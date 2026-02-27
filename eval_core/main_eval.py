# main_eval.py
# -*- coding: utf-8 -*-
"""
main_eval.py  (ENTRYPOINT, config-driven)
-----------------------------------------
融合版推理入口：加载 bundle -> preprocess -> encode -> route -> risk controls -> (optional explain) -> 输出

所有阈值与开关由 config.py 驱动：
- cfg["risk"]["tau_risk"], cfg["risk"]["tau_epi"]
- cfg["explain"]["enable"], cfg["explain"]["top_k"]

命令行只负责：
- --config_path: 指定 JSON 配置文件
- --model_root / --input_path / --output_path / --save_json: 路径覆盖（可选）
- --override: 简单 key=value 覆盖（可选）
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np

from config import get_default_config, load_config

from preprocess import preprocess_raw_sample, get_report_type, read_json, read_jsonl, write_text
from encoder import encode
from bundle import load_bundle, pick_expert_key
from risk_model import predict_with_risk_controls
from explain import build_evidence_chain, topk_contrib_softmax_target


# =========================================================
# IO
# =========================================================

def load_input(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input not found: {path}")

    # 1) jsonl
    if path.lower().endswith(".jsonl"):
        data = read_jsonl(path)
        if not data:
            raise ValueError("Empty jsonl input")
        return data

    obj = read_json(path)

    # 2) list
    if isinstance(obj, list):
        return obj

    # 3) dict wrappers
    if isinstance(obj, dict):
        # original supported format
        if isinstance(obj.get("dataset", None), list):
            return obj["dataset"]

        # ✅ your fixed format: {"info": [ {sample0}, {sample1}, ... ], ...}
        # We convert it to: [ {"info":[sample0]}, {"info":[sample1]}, ... ]
        if isinstance(obj.get("info", None), list) and len(obj["info"]) > 0 and isinstance(obj["info"][0], dict):
            return [{"info": [it]} for it in obj["info"]]

    raise ValueError("Unsupported input json format. Expect list or {'dataset': list} or jsonl or {'info': list}.")


def _fmt_vec(v: np.ndarray, k: int = 4) -> str:
    v = np.asarray(v, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{x:.{k}f}" for x in v) + "]"

def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for s in pairs:
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()

        vv: Any
        if v.lower() in ("true", "false"):
            vv = (v.lower() == "true")
        else:
            try:
                if "." in v:
                    vv = float(v)
                else:
                    vv = int(v)
            except Exception:
                try:
                    vv = float(v)
                except Exception:
                    vv = v

        parts = k.split(".")
        cur = out
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = vv
    return out

def deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in new.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


# =========================================================
# Main
# =========================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--config_path", type=str, default="", help="可选：JSON 配置文件路径")
    ap.add_argument("--model_root", type=str, default="", help="可选：覆盖 cfg.global.model_root")
    ap.add_argument("--processed_report_info_path", type=str, default="", help="可选：覆盖 cfg.paths.processed_report_info_path")
    ap.add_argument("--eval_info_path", type=str, default="", help="可选：覆盖 cfg.paths.eval_info_path")
    ap.add_argument("--save_json", type=str, default="", help="可选：保存逐样本结果 json")
    ap.add_argument("--override", type=str, nargs="*", default=[], help="可选：覆盖配置项，如 risk.tau_risk=0.25 explain.enable=true")
    args = ap.parse_args()

    # 0) config
    cfg = get_default_config()
    if args.config_path:
        cfg = load_config(args.config_path, base_cfg=cfg)
    if args.override:
        cfg = deep_update(cfg, parse_overrides(args.override))

    script_dir = os.path.dirname(os.path.abspath(__file__))

    default_model_root = os.path.join(script_dir, "model")
    model_root = args.model_root or default_model_root
    cfg.setdefault("global", {})
    cfg["global"]["model_root"] = model_root

    cfg.setdefault("paths", {})
    input_path = args.processed_report_info_path
    
    output_path = args.eval_info_path
    
    save_json_path = args.save_json


    cfg["paths"]["input_path"] = input_path
    cfg["paths"]["output_path"] = output_path
    if save_json_path:
        cfg["paths"]["save_json"] = save_json_path

    # thresholds / switches
    tau_risk = float(cfg["risk"]["tau_risk"])
    tau_epi = float(cfg["risk"]["tau_epi"])
    do_explain = bool(cfg["explain"]["enable"])
    top_k = int(cfg["explain"]["top_k"])

    # 1) load bundle
    bundle = load_bundle(model_root)
    encoder = bundle.encoder

    # 2) load input
    dataset = load_input(input_path)
    if len(dataset) == 0:
        raise ValueError("Empty input dataset")

    lines: List[str] = []
    json_out: List[Dict[str, Any]] = []

    lines.append("=== Fusion Inference Report (config-driven) ===")
    lines.append(f"model_root: {model_root}")
    lines.append(f"input_path: {input_path}")
    lines.append(f"n_samples: {len(dataset)}")
    lines.append(f"risk: tau_risk={tau_risk}, tau_epi={tau_epi}")
    lines.append(f"explain: enable={do_explain}, top_k={top_k}")
    lines.append("")

    for idx, sample in enumerate(dataset):
        rt = int(get_report_type(sample, default=-1))
        expert_key = pick_expert_key(rt, bundle)
        expert = bundle.experts[expert_key]

        x32 = preprocess_raw_sample(sample)  # (1,32)
        z = encode(encoder, x32)[0]          # (d,)

        out = predict_with_risk_controls(
            z=z,
            head=expert.head,
            calibrator=expert.calibrator,
            conformal=expert.conformal,
            tau_risk=tau_risk,
            tau_epi=tau_epi,
        )

        # explanation
        contribs = []
        if do_explain and expert.head.W_map is not None:
            try:
                contribs = topk_contrib_softmax_target(
                    x=z,
                    W=expert.head.W_map,
                    p=out["p_final"],
                    target_labels=[1, 2],
                    feature_names=None,
                    top_k=top_k,
                )
            except Exception as e:
                contribs = [{"feature": "contrib_error", "error": str(e)}]

        p_model = {
            "route": {"report_type": rt, "expert_key": expert_key, "expert_kind": expert.kind},
            "pred": int(out["pred"]),
            "pred_set": list(map(int, out["pred_set"])),
            "abstain": bool(out["abstain"]),
            "p_final": out["p_final"].tolist(),
            "p_abnormal": float(out["p_abnormal"]),
            "p_severe": float(out["p_severe"]),
            "entropy": float(out["entropy"]),
            "epistemic": float(out["epistemic"]),
            "aleatoric": float(out["aleatoric"]),
        }

        evidence = build_evidence_chain(
            p_model=p_model,
            lf_names=None,
            lf_outputs=None,
            quality_severity=None,
            multiclass_x=z if do_explain else None,
            multiclass_W=expert.head.W_map if (do_explain and expert.head.W_map is not None) else None,
            multiclass_p=np.asarray(out["p_final"]) if do_explain else None,
            target_labels=[1, 2] if do_explain else None,
            feature_names=None,
            top_k=top_k,
        )

        # text output
        lines.append(f"--- Sample {idx} ---")
        lines.append(f"report_type: {rt}")
        lines.append(f"expert: {expert_key} ({expert.kind})")
        lines.append(f"p_final: {_fmt_vec(out['p_final'], k=4)}")
        lines.append(f"pred: {out['pred']} | pred_set: {out['pred_set']} | abstain: {out['abstain']}")
        lines.append(f"p_abnormal(p1+p2): {out['p_abnormal']:.4f} | p_severe(p2): {out['p_severe']:.4f}")
        lines.append(f"uncertainty: entropy={out['entropy']:.4f}, epistemic={out['epistemic']:.6f}, aleatoric={out['aleatoric']:.4f}")

        if do_explain:
            lines.append("top_feature_contributions (embedding):")
            for c in evidence.get("top_feature_contributions", [])[:top_k]:
                feat = c.get("feature", "")
                contrib = c.get("contribution", None)
                grad = c.get("grad", None)
                xv = c.get("x", None)
                if contrib is None:
                    lines.append(f"  - {feat}: {c}")
                else:
                    lines.append(f"  - {feat}: x={xv:.4f}, grad={grad:.6f}, contrib={contrib:.6f}")
        lines.append("")

        json_out.append({
            "index": idx,
            "report_type": rt,
            "expert_key": expert_key,
            "expert_kind": expert.kind,
            "result": out,
            "evidence": evidence,
        })

    write_text(output_path, "\n".join(lines), encoding="utf-8")
    print("[Saved]", output_path)

    if save_json_path:
        os.makedirs(os.path.dirname(save_json_path) or ".", exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(json_out, f, ensure_ascii=False, indent=2)
        print("[Saved]", save_json_path)


if __name__ == "__main__":
    main()
