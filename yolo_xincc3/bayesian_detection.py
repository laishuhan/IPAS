import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
from tqdm import tqdm
from enum import Enum


# ============================================================================
# 第一部分：几何计算工具函数
# ============================================================================

def ensure_ccw(vertices: np.ndarray) -> np.ndarray:
    """确保多边形顶点按逆时针顺序排列"""
    n = len(vertices)
    if n < 3:
        return vertices
    
    signed_area = 0.0
    for i in range(n):
        j = (i + 1) % n
        signed_area += vertices[i, 0] * vertices[j, 1]
        signed_area -= vertices[j, 0] * vertices[i, 1]
    
    if signed_area < 0:
        return vertices[::-1].copy()
    return vertices


def polygon_area(vertices: np.ndarray) -> float:
    """计算多边形面积（Shoelace公式）"""
    n = len(vertices)
    if n < 3:
        return 0.0
    
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    
    return abs(area) / 2.0


def polygon_intersection(poly1: np.ndarray, poly2: np.ndarray) -> np.ndarray:
    """计算两个凸多边形的交集（Sutherland-Hodgman算法）"""
    def inside_edge(point, edge_start, edge_end):
        cross = ((edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) -
                 (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0]))
        return cross >= 0
    
    def compute_intersection(s, e, edge_start, edge_end):
        d1 = e - s
        d2 = edge_end - edge_start
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-10:
            return s
        t = ((edge_start[0] - s[0]) * d2[1] - (edge_start[1] - s[1]) * d2[0]) / cross
        return s + t * d1
    
    output = poly1.copy()
    
    for i in range(len(poly2)):
        if len(output) == 0:
            break
        
        input_poly = output
        output = []
        
        edge_start = poly2[i]
        edge_end = poly2[(i + 1) % len(poly2)]
        
        for j in range(len(input_poly)):
            current = input_poly[j]
            next_vertex = input_poly[(j + 1) % len(input_poly)]
            
            curr_in = inside_edge(current, edge_start, edge_end)
            next_in = inside_edge(next_vertex, edge_start, edge_end)
            
            if curr_in:
                if next_in:
                    output.append(next_vertex)
                else:
                    output.append(compute_intersection(current, next_vertex, edge_start, edge_end))
            elif next_in:
                output.append(compute_intersection(current, next_vertex, edge_start, edge_end))
                output.append(next_vertex)
        
        output = np.array(output) if output else np.array([])
    
    return np.array(output) if len(output) > 0 else np.array([])


def calculate_obb_iou(obb1: np.ndarray, obb2: np.ndarray) -> float:
    """计算两个OBB的IoU"""
    obb1 = np.array(obb1).reshape(4, 2)
    obb2 = np.array(obb2).reshape(4, 2)
    
    obb1 = ensure_ccw(obb1)
    obb2 = ensure_ccw(obb2)
    
    area1 = polygon_area(obb1)
    area2 = polygon_area(obb2)
    
    if area1 < 1e-6 or area2 < 1e-6:
        return 0.0
    
    intersection = polygon_intersection(obb1, obb2)
    
    if len(intersection) < 3:
        return 0.0
    
    inter_area = polygon_area(intersection)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 1e-6 else 0.0


def obb_to_xyxy(obb: np.ndarray) -> np.ndarray:
    """将OBB四角点转换为轴对齐边界框"""
    obb = np.array(obb).reshape(-1, 2)
    return np.array([np.min(obb[:, 0]), np.min(obb[:, 1]),
                     np.max(obb[:, 0]), np.max(obb[:, 1])])





# ============================================================================
# 第二部分：Dropout注入器
# ============================================================================
class DropoutInjector:
    def __init__(self, model, dropout_rate: float = 0.15):
        """
        Args:
            model: YOLO模型
            dropout_rate: dropout概率
        """
        self.model = model
        self.dropout_rate = dropout_rate
        self.hooks = []
        self.dropout_layers = []
        self.enabled = False
        
        # 指定要添加dropout的层
        self.target_layers = {
            # Backbone深层特征
            'model.model.6',   # P4的C3k2输出  
            'model.model.8',   # P5的C3k2输出
            'model.model.9',   # SPPF输出
            'model.model.10',  # C2PSA输出
            
            # Head特征融合层
            'model.model.13',  # P4 upsample融合后
            'model.model.16',  # P3 upsample融合后
            'model.model.19',  # P4 downsample融合后
            'model.model.22',  # P5 downsample融合后
        }
        
        self.device = next(model.model.parameters()).device
        self._register_hooks()
        if not self.hooks:
            raise RuntimeError(
                "No dropout hooks registered. Check target_layers for the current model."
            )

    
    def _register_hooks(self):
        """只在指定层注册hooks"""
        inner_model = self.model.model if hasattr(self.model, 'model') else self.model
        self._register_recursive(inner_model, 'model')
    
    def _register_recursive(self, module: nn.Module, prefix: str):
        """递归遍历，精确匹配目标层"""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # 精确匹配：只在完整路径匹配时添加
            if full_name in self.target_layers:
                self._add_dropout_hook(child)
            
            self._register_recursive(child, full_name)
    
    def _add_dropout_hook(self, module: nn.Module):
        """为指定模块添加dropout hook"""
        dropout = nn.Dropout2d(p=self.dropout_rate).to(self.device)
        dropout.train()
        idx = len(self.dropout_layers)
        self.dropout_layers.append(dropout)
        
        def make_hook(dropout_idx):
            def hook_fn(module, input, output):
                if self.enabled:
                    self.dropout_layers[dropout_idx].train()
                    return self.dropout_layers[dropout_idx](output)
                return output
            return hook_fn
        
        handle = module.register_forward_hook(make_hook(idx))
        self.hooks.append(handle)
    
    def enable(self):
        """启用dropout"""
        self.enabled = True
        for d in self.dropout_layers:
            d.train()
    
    def disable(self):
        """禁用dropout"""
        self.enabled = False
        for d in self.dropout_layers:
            d.eval()
    
    def remove_hooks(self):
        """移除所有hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.dropout_layers.clear()


# ============================================================================
# 第三部分：贝叶斯YOLO主类
# ============================================================================

class BayesianYOLO:
    """贝叶斯YOLO推理器 - OBB专用版本"""
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'cuda',
                 n_samples: int = 30,
                 dropout_rate: float = 0.1,
                 iou_threshold: float = 0.3,
                 min_samples: int = 5):
        
        self.device = device
        self.n_samples = n_samples
        self.iou_threshold = iou_threshold
        self.min_samples = min_samples
        self.model = YOLO(model_path, task='obb')
        self.model.to(device)
        
        self.dropout_injector = DropoutInjector(self.model, dropout_rate)
        
    


    def predict_with_uncertainty(
        self,
        source,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        tta_sources=None,
        **kwargs
    ) -> Tuple[List[Dict], Dict]:
        if tta_sources is None:
            if isinstance(source, (list, tuple)):
                sources = list(source)
            else:
                sources = [source]
        else:
            sources = list(tta_sources)

        total_samples = self.n_samples * len(sources)
        all_detections = []
        self.model.model.eval()
        self.dropout_injector.enable()

        # Debug: track confidence variation
        first_det_confs = []

        for idx in range(total_samples):
            src = sources[idx // self.n_samples]
            try:
                results = self.model.predict(
                    source=src, conf=conf, iou=iou,
                    imgsz=imgsz, verbose=False, **kwargs
                )

                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
                        dets = self._extract_detections(result)
                        all_detections.append(dets)
                        if dets:
                            first_det_confs.append(dets[0]['conf'])
                    else:
                        all_detections.append([])
                else:
                    all_detections.append([])
            except Exception:
                all_detections.append([])

        self.dropout_injector.disable()

        return self._aggregate_detections(all_detections)

    def _extract_detections(self, result) -> List[Dict]:
        dets = []
        obb = result.obb
        corners = obb.xyxyxyxy.cpu().numpy()
        confs = obb.conf.cpu().numpy()
        classes = obb.cls.cpu().numpy()
        
        for i in range(len(corners)):
            dets.append({
                'obb': corners[i],
                'conf': float(confs[i]),
                'cls': int(classes[i])
            })
        return dets
    
    def _aggregate_detections(self, all_detections: List[List[Dict]]) -> Tuple[List[Dict], Dict]:
        # 展平
        all_dets = []
        for sample_idx, dets in enumerate(all_detections):
            for det in dets:
                det['sample_idx'] = sample_idx
                all_dets.append(det)
        
        if not all_dets:
            return [], {
                'avg_conf_std': 0,
                'num_detections_std': 0,
                'detection_stability': 1,
                'raw_detections': 0,
                'cluster_count': 0,
                'final_detections': 0,
                'clusters': []
            }
        
        # 按置信度排序
        all_dets.sort(key=lambda x: x['conf'], reverse=True)
        
        # 聚类
        clusters = []
        used = [False] * len(all_dets)
        
        for i, det in enumerate(all_dets):
            if used[i]:
                continue
            
            cluster = {
                'obbs': [det['obb']],
                'confs': [det['conf']],
                'cls': det['cls'],
                'samples': {det['sample_idx']}
            }
            used[i] = True
            
            for j in range(i + 1, len(all_dets)):
                if used[j] or all_dets[j]['cls'] != det['cls']:
                    continue
                
                iou = calculate_obb_iou(det['obb'], all_dets[j]['obb'])
                if iou >= self.iou_threshold:
                    cluster['obbs'].append(all_dets[j]['obb'])
                    cluster['confs'].append(all_dets[j]['conf'])
                    cluster['samples'].add(all_dets[j]['sample_idx'])
                    used[j] = True
            
            clusters.append(cluster)
        
        # 统计
        final = []
        cluster_summaries = []
        conf_stds = []
        
        for cluster_id, c in enumerate(clusters, start=1):
            n = len(c['samples'])
            obbs = np.array(c['obbs'])
            confs = np.array(c['confs'])
            
            summary = {
                'cluster_id': cluster_id,
                'obb': np.mean(obbs, axis=0),
                'conf': np.mean(confs),
                'conf_std': np.std(confs),
                'obb_std': np.mean(np.std(obbs, axis=0)),
                'cls': c['cls'],
                'num_samples': n
            }
            cluster_summaries.append(summary)
            
            if n < self.min_samples:
                continue
            final.append(summary.copy())
            conf_stds.append(np.std(confs))
        
        final.sort(key=lambda x: x['conf'], reverse=True)
        
        det_counts = [len(d) for d in all_detections]
        uncertainties = {
            'avg_conf_std': np.mean(conf_stds) if conf_stds else 0.0,
            'num_detections_std': np.std(det_counts),
            'detection_stability': 1.0 / (1.0 + np.std(det_counts)),
            'raw_detections': len(all_dets),
            'cluster_count': len(clusters),
            'final_detections': len(final),
            'clusters': cluster_summaries
        }
        
        return final, uncertainties
    
    def __del__(self):
        if hasattr(self, 'dropout_injector'):
            self.dropout_injector.remove_hooks()




# ============================================================================
# 第四部分：不确定性分析器
# ============================================================================

class ConfidenceLevel(Enum):
    """置信度等级"""
    HIGH = "高置信度"
    MEDIUM = "中置信度"
    LOW = "低置信度"
    VERY_LOW = "极低置信度"


class UncertaintyAnalyzer:
    """
    多维度不确定性分析器
    
    评估维度：
    1. 绝对置信度 (conf)
    2. 相对不确定性 (conf_std / conf)  
    3. 检测稳定性 (num_samples / total_samples)
    4. 位置不确定性 (obb_std)
    """
    
    def __init__(self, 
                 total_samples: int = 30,
                 conf_threshold_high: float = 0.75,
                 conf_threshold_medium: float = 0.55,
                 relative_std_threshold_high: float = 0.05,
                 relative_std_threshold_low: float = 0.15,
                 stability_threshold: float = 0.60):
        """
        Args:
            total_samples: MC采样总次数
            conf_threshold_high: 高置信度阈值（绝对值）
            conf_threshold_medium: 中置信度阈值（绝对值）
            relative_std_threshold_high: 相对不确定性高阈值
            relative_std_threshold_low: 相对不确定性低阈值
            stability_threshold: 稳定性阈值（出现率）
        """
        self.total_samples = total_samples
        self.conf_high = conf_threshold_high
        self.conf_medium = conf_threshold_medium
        self.rel_std_high = relative_std_threshold_high
        self.rel_std_low = relative_std_threshold_low
        self.stability_threshold = stability_threshold
    
    def calculate_quality_score(self, det: Dict) -> Tuple[float, Dict]:
        """
        计算综合质量分数 (0-100)
        
        Returns:
            quality_score: 综合分数
            components: 各维度得分详情
        """
        # 1. 绝对置信度得分 (0-35, 指数饱和)
        conf_score = 35 * (1 - np.exp(-det['conf'] / 0.25))
        
        # 2. 相对不确定性得分 (0-20, 越稳定越高)
        relative_std = det['conf_std'] / max(det['conf'], 0.1)
        uncertainty_score = 20 * np.exp(-relative_std / 0.15)
        
        # 3. 检测稳定性得分 (0-35, 指数饱和)
        stability = det['num_samples'] / self.total_samples
        stability_score = 35 * (1 - np.exp(-stability / 0.25))
        # 轻度惩罚：低出现率时略微削弱置信度分
        stability_factor = 0.8 + 0.2 * min(1.0, stability / 0.5)
        conf_score *= stability_factor
        
        # 4. 位置精度得分 (0-10)
        obb_std = det.get('obb_std', 0)
        position_score = 10 * np.exp(-obb_std / 3.0)
        
        # 综合分数
        total_score = conf_score + uncertainty_score + stability_score + position_score
        
        components = {
            'conf_score': conf_score,
            'uncertainty_score': uncertainty_score,
            'stability_score': stability_score,
            'position_score': position_score,
            'relative_std': relative_std,
            'stability': stability,
            'stability_factor': stability_factor
        }
        
        return total_score, components
    
    def classify_detection(self, det: Dict) -> Tuple[ConfidenceLevel, str]:
        """
        综合分类检测质量
        
        Returns:
            level: 置信度等级
            reason: 分类原因
        """
        quality_score, components = self.calculate_quality_score(det)
        
        conf = det['conf']
        relative_std = components['relative_std']
        stability = components['stability']
        
        # 规则1: 极低置信度 - 无论其他指标如何
        if conf < 0.50:
            return ConfidenceLevel.VERY_LOW, f"绝对置信度过低({conf:.3f})"
        
        # 规则2: 不稳定检测 - 出现率低
        if stability < self.stability_threshold:
            return ConfidenceLevel.LOW, f"检测不稳定(出现率{stability:.1%})"
        
        # 规则3: 高相对不确定性
        if relative_std > self.rel_std_low:
            if conf > self.conf_high:
                return ConfidenceLevel.MEDIUM, f"置信度高但波动大(相对std={relative_std:.3f})"
            else:
                return ConfidenceLevel.LOW, f"相对不确定性过高({relative_std:.3f})"
        
        # 规则4: 综合评分
        if quality_score >= 80:
            return ConfidenceLevel.HIGH, f"综合质量优秀(分数={quality_score:.1f})"
        elif quality_score >= 65:
            return ConfidenceLevel.MEDIUM, f"综合质量良好(分数={quality_score:.1f})"
        elif quality_score >= 50:
            return ConfidenceLevel.LOW, f"综合质量一般(分数={quality_score:.1f})"
        else:
            return ConfidenceLevel.VERY_LOW, f"综合质量较差(分数={quality_score:.1f})"
    
    def analyze(self, detections: List[Dict], uncertainties: Dict) -> Dict:
        """
        主分析方法
        """
        analysis = {
            'high_confidence': [],
            'medium_confidence': [],
            'low_confidence': [],
            'very_low_confidence': [],
            'statistics': {},
            'recommendations': [],
            'detailed_scores': []
        }
        
        for idx, det in enumerate(detections):
            # 计算质量分数和分类
            quality_score, components = self.calculate_quality_score(det)
            level, reason = self.classify_detection(det)
            
            # 构建详细信息
            info = {
                'idx': idx,
                'cls': det['cls'],
                'conf': det['conf'],
                'conf_std': det.get('conf_std', 0),
                'num_samples': det['num_samples'],
                'quality_score': quality_score,
                'level': level.value,
                'reason': reason,
                'components': components
            }
            
            # 分类
            if level == ConfidenceLevel.HIGH:
                analysis['high_confidence'].append(info)
            elif level == ConfidenceLevel.MEDIUM:
                analysis['medium_confidence'].append(info)
            elif level == ConfidenceLevel.LOW:
                analysis['low_confidence'].append(info)
            else:
                analysis['very_low_confidence'].append(info)
            
            analysis['detailed_scores'].append(info)
        
        # 计算统计
        total = len(detections)
        analysis['statistics'] = {
            'total_detections': total,
            'high_confidence_count': len(analysis['high_confidence']),
            'medium_confidence_count': len(analysis['medium_confidence']),
            'low_confidence_count': len(analysis['low_confidence']),
            'very_low_confidence_count': len(analysis['very_low_confidence']),
            'high_confidence_ratio': len(analysis['high_confidence']) / max(1, total),
            'avg_quality_score': np.mean([d['quality_score'] for d in analysis['detailed_scores']]) if detections else 0,
            'avg_confidence_std': uncertainties.get('avg_conf_std', 0),
            'detection_stability': uncertainties.get('detection_stability', 1),
            'raw_detections': uncertainties.get('raw_detections'),
            'cluster_count': uncertainties.get('cluster_count'),
            'final_detections': uncertainties.get('final_detections', total)
        }
        
        # 生成建议
        self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict):
        """生成智能建议"""
        stats = analysis['statistics']
        recommendations = []
        
        if stats['total_detections'] == 0:
            recommendations.append("⚠️ 未检测到目标")
            recommendations.append("建议: 降低conf阈值或检查图像质量")
        else:
            # 质量分析
            high_ratio = stats['high_confidence_ratio']
            avg_score = stats['avg_quality_score']
            
            if high_ratio >= 0.8 and avg_score >= 75:
                recommendations.append("✅ 整体质量优秀，可自动化处理")
            elif high_ratio >= 0.6 and avg_score >= 65:
                recommendations.append("✓ 整体质量良好，建议抽检20%")
            elif high_ratio >= 0.4:
                recommendations.append("⚠️ 质量一般，建议人工复核50%")
            else:
                recommendations.append("🔴 质量较差，需要全面人工复核")
            
            # 具体问题
            if stats['very_low_confidence_count'] > 0:
                recommendations.append(f"🔴 {stats['very_low_confidence_count']}个极低质量检测，应删除")
            
            if stats['low_confidence_count'] > 0:
                recommendations.append(f"⚠️ {stats['low_confidence_count']}个低质量检测，需人工确认")
            
            # 不稳定性警告
            low_stability = [d for d in analysis['detailed_scores'] 
                           if d['components']['stability'] < 0.5]
            if low_stability:
                recommendations.append(f"⚠️ {len(low_stability)}个检测出现率<50%，可能是假阳性")
            
            # 高波动警告
            high_variation = [d for d in analysis['detailed_scores']
                            if d['components']['relative_std'] > 0.15]
            if high_variation:
                recommendations.append(f"⚠️ {len(high_variation)}个检测波动过大，建议复核")
        
        analysis['recommendations'] = recommendations
    

    def build_report(self, analysis: Dict) -> str:
        """组装详细报告文本"""
        lines = []
        lines.append("=" * 80)
        lines.append("多维度不确定性分析报告")
        lines.append("=" * 80)
        lines.append("")

        stats = analysis['statistics']
        lines.append("【总体统计】")
        raw_detections = stats.get('raw_detections')
        cluster_count = stats.get('cluster_count')
        final_detections = stats.get('final_detections', stats['total_detections'])
        if raw_detections is not None:
            lines.append(f"  总检测数: {raw_detections}")
        if cluster_count is not None:
            lines.append(f"  聚类数: {cluster_count}")
        lines.append(f"  最终检测数: {final_detections}")
        lines.append(f"  平均质量分数: {stats['avg_quality_score']:.1f}/100")
        lines.append(f"  平均conf_std: {stats['avg_confidence_std']:.4f}")
        lines.append(f"  检测稳定性: {stats['detection_stability']:.4f}")
        lines.append("")

        lines.append("【质量分布】")
        lines.append(f"  ⭐⭐⭐ 高置信度: {stats['high_confidence_count']} ({stats['high_confidence_ratio']:.1%})")
        lines.append(f"  ⭐⭐   中置信度: {stats['medium_confidence_count']}")
        lines.append(f"  ⭐     低置信度: {stats['low_confidence_count']}")
        lines.append(f"  ❌     极低置信度: {stats['very_low_confidence_count']}")

        for level_name, level_key, emoji in [
            ("高置信度检测", "high_confidence", "⭐⭐⭐"),
            ("中置信度检测", "medium_confidence", "⭐⭐"),
            ("低置信度检测", "low_confidence", "⭐"),
            ("极低置信度检测", "very_low_confidence", "❌"),
        ]:
            dets = analysis[level_key]
            if dets:
                lines.append("")
                lines.append(f"【{level_name}】{emoji}")
                for d in dets:
                    lines.append(
                        f"  #{d['idx']}: conf={d['conf']:.3f}, "
                        f"std={d['conf_std']:.4f}, "
                        f"出现率={d['components']['stability']:.1%}, "
                        f"质量={d['quality_score']:.1f}"
                    )
                    lines.append(f"         原因: {d['reason']}")

        lines.append("=" * 80)
        return "\n".join(lines)

    def export_review_list(self, analysis: Dict, output_file: str = None):
        """导出需要人工复核的检测列表"""
        review_list = []
        
        # 优先级1: 极低质量
        for d in analysis['very_low_confidence']:
            review_list.append({
                'idx': d['idx'],
                'priority': 'HIGH',
                'conf': d['conf'],
                'quality_score': d['quality_score'],
                'reason': d['reason'],
                'action': 'DELETE'
            })
        
        # 优先级2: 低质量
        for d in analysis['low_confidence']:
            review_list.append({
                'idx': d['idx'],
                'priority': 'MEDIUM',
                'conf': d['conf'],
                'quality_score': d['quality_score'],
                'reason': d['reason'],
                'action': 'REVIEW'
            })
        
        # 优先级3: 中等质量（选择性）
        for d in analysis['medium_confidence']:
            if d['quality_score'] < 70:
                review_list.append({
                    'idx': d['idx'],
                    'priority': 'LOW',
                    'conf': d['conf'],
                    'quality_score': d['quality_score'],
                    'reason': d['reason'],
                    'action': 'OPTIONAL_REVIEW'
                })
        
        # 按优先级和质量分数排序
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        review_list.sort(key=lambda x: (priority_order[x['priority']], x['quality_score']))
        
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(review_list, f, indent=2)
        
        return review_list

# 设置 matplotlib 的中文字体支持
def setup_font():
    import matplotlib.font_manager as fm
    fonts = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
    return None


class BayesianYOLOInference:
    def __init__(self, model_path: str, device: str = 'cuda',
                 n_samples: int = 30, dropout_rate: float = 0.2,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        
        self.n_samples = n_samples
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        setup_font()
        
        self.bayesian_model = BayesianYOLO(
            model_path=model_path,
            device=device,
            n_samples=n_samples,
            dropout_rate=dropout_rate,
            iou_threshold=iou_threshold,
            min_samples=7,
        )
        
        self.analyzer = UncertaintyAnalyzer(total_samples=n_samples)
    
    def inference(
        self,
        image_path: str,
        output_dir: str = './results',
        predict_dir: str = None
    ) -> Tuple[List[Dict], Dict, Dict]:

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        tta_images = self._generate_tta_images(image)
        total_samples = self.n_samples * len(tta_images)
        self.analyzer.total_samples = total_samples

        detections, uncertainties = self.bayesian_model.predict_with_uncertainty(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=960,
            tta_sources=tta_images
        )
        
        analysis = self.analyzer.analyze(detections, uncertainties)
        analysis_report = self.analyzer.build_report(analysis)
        
        visual_dir = output_dir
        if predict_dir:
            visual_dir = os.path.join(predict_dir, 'report')
        self._save_results(image_rgb, detections, uncertainties,
                           visual_dir, os.path.basename(image_path),
                           analysis_report=analysis_report,
                           total_samples=total_samples)
        if predict_dir:
            self._save_predict_outputs(image_path, image, detections, analysis, predict_dir)
        
        return detections, uncertainties, analysis
    
    def _save_results(
        self,
        image,
        detections,
        uncertainties,
        output_dir,
        name,
        analysis_report=None,
        total_samples=None
    ):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._save_report(detections, uncertainties, output_dir, name, analysis_report, total_samples)

    def _save_predict_outputs(self, image_path, image, detections, analysis, output_dir):
        output_path = Path(output_dir)
        labels_dir = output_path / 'labels'
        labels_visual_dir = output_path / 'visual'
        output_path.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        labels_visual_dir.mkdir(parents=True, exist_ok=True)

        image_name = Path(image_path).name
        target_image = output_path / image_name
        try:
            shutil.copy2(image_path, target_image)
        except Exception:
            cv2.imwrite(str(target_image), image)

        label_name = f"{Path(image_path).stem}.txt"
        label_path = labels_dir / label_name

        selected_indices = {
            info['idx']
            for info in (analysis.get('high_confidence', []) + analysis.get('medium_confidence', []))
        }

        with open(label_path, 'w', encoding='utf-8') as f:
            for idx, det in enumerate(detections):
                if idx not in selected_indices:
                    continue
                line = self._format_obb_label(det, image.shape)
                if line:
                    f.write(line + "\n")

        self._save_label_visual(image, label_path, labels_visual_dir, image_name)

    def _format_obb_label(self, det, image_shape):
        obb = np.array(det['obb']).reshape(-1, 2).astype(float)
        h, w = image_shape[:2]
        if w <= 0 or h <= 0:
            return ""

        obb[:, 0] /= w
        obb[:, 1] /= h
        coords = obb.reshape(-1)
        cls_id = int(det.get('cls', 0))
        conf = float(det.get('conf', 0.0))

        parts = [str(cls_id)]
        parts.extend([f"{v:.6f}" for v in coords])
        parts.append(f"{conf:.6f}")
        return " ".join(parts)

    def _class_color(self, cls_id: int) -> Tuple[int, int, int]:
        palette = [
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]
        return palette[int(cls_id) % len(palette)]

    def _save_label_visual(self, image, label_path: Path, output_dir: Path, image_name: str):
        if not label_path.exists():
            return
        h, w = image.shape[:2]
        if w <= 0 or h <= 0:
            return

        visual = image.copy()
        lines = label_path.read_text(encoding='utf-8').splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            try:
                cls_id = int(float(parts[0]))
                values = [float(v) for v in parts[1:]]
            except ValueError:
                continue
            if len(values) < 8:
                continue

            coords = values[:8]
            conf = values[8] if len(values) >= 9 else None
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h

            color = self._class_color(cls_id)
            pts_int = np.round(pts).astype(int).reshape(-1, 1, 2)
            cv2.polylines(visual, [pts_int], True, color, 2, lineType=cv2.LINE_AA)

            label = f"{cls_id}"
            if conf is not None:
                label = f"{cls_id}:{conf:.2f}"
            x, y = pts_int[0, 0]
            y = max(0, y - 4)
            cv2.putText(
                visual,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        output_path = output_dir / image_name
        cv2.imwrite(str(output_path), visual)

    def _generate_tta_images(self, image):
        # Conservative pixel-only TTA (no coordinate transforms).
        images = [image]
        images.append(self._brightness_add(image, 8))
        images.append(self._brightness_mul(image, 1.03))
        images.append(self._contrast(image, 1.03))
        images.append(self._gamma(image, 1.05))
        images.append(self._saturation(image, 1.03))
        images.append(self._sharpen(image, 0.2))
        images.append(self._gaussian_noise(image, 3.0))
        return images

    def _brightness_add(self, image, delta):
        out = image.astype(np.float32) + float(delta)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _brightness_mul(self, image, factor):
        out = image.astype(np.float32) * float(factor)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _contrast(self, image, factor):
        out = (image.astype(np.float32) - 128.0) * float(factor) + 128.0
        return np.clip(out, 0, 255).astype(np.uint8)

    def _gamma(self, image, gamma):
        inv_gamma = 1.0 / float(gamma)
        table = (np.arange(256) / 255.0) ** inv_gamma * 255.0
        table = np.clip(table, 0, 255).astype(np.uint8)
        return cv2.LUT(image, table)

    def _saturation(self, image, factor):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(factor), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _sharpen(self, image, amount):
        blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
        return cv2.addWeighted(image, 1.0 + float(amount), blur, -float(amount), 0)

    def _gaussian_noise(self, image, sigma):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, float(sigma), image.shape).astype(np.float32)
        out = image.astype(np.float32) + noise
        return np.clip(out, 0, 255).astype(np.uint8)
    
    def _visualize(self, image, detections, uncertainties, output_dir, name):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        ax1 = axes[0]
        ax1.imshow(image)
        ax1.set_title('Detection Results', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        cluster_summaries = None
        if uncertainties:
            cluster_summaries = uncertainties.get('clusters')

        if cluster_summaries:
            cmap = plt.cm.get_cmap('tab20', max(1, len(cluster_summaries)))
            for idx, cluster in enumerate(cluster_summaries):
                obb = np.array(cluster['obb']).reshape(-1, 2)
                color = cmap(idx)

                polygon = Polygon(obb, linewidth=2, edgecolor=color, facecolor='none')
                ax1.add_patch(polygon)

                score, _ = self.analyzer.calculate_quality_score(cluster)
                label_x, label_y = np.min(obb[:, 0]), np.min(obb[:, 1]) - 5
                cluster_id = cluster.get('cluster_id', idx + 1)
                label = f'ID:{cluster_id} N:{cluster["num_samples"]} Q:{score:.1f}'
                ax1.text(label_x, label_y, label, fontsize=8, color='white',
                        fontweight='bold', bbox=dict(boxstyle='round,pad=0.2',
                        facecolor=color, alpha=0.7))
        elif detections:
            for idx, det in enumerate(detections):
                obb = det['obb']
                conf = det['conf']
                conf_std = det.get('conf_std', 0)
                
                color = 'green' if conf_std < 0.02 else ('orange' if conf_std < 0.05 else 'red')
                
                polygon = Polygon(obb, linewidth=2, edgecolor=color, facecolor='none')
                ax1.add_patch(polygon)
                
                label_x, label_y = np.min(obb[:, 0]), np.min(obb[:, 1]) - 5
                label = f'#{idx+1} C:{conf:.2f} S:{conf_std:.3f}'
                ax1.text(label_x, label_y, label, fontsize=8, color='white',
                        fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', 
                        facecolor=color, alpha=0.8))
        else:
            ax1.text(0.5, 0.5, 'No Detection', transform=ax1.transAxes,
                    fontsize=16, ha='center', va='center',
                    bbox=dict(facecolor='yellow', alpha=0.7))
        
        ax2 = axes[1]
        ax2.imshow(image)
        ax2.set_title('Uncertainty Heatmap', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        if detections:
            heatmap = np.zeros(image.shape[:2], dtype=np.float32)
            for det in detections:
                xyxy = obb_to_xyxy(det['obb']).astype(int)
                x1, y1 = max(0, xyxy[0]), max(0, xyxy[1])
                x2, y2 = min(image.shape[1], xyxy[2]), min(image.shape[0], xyxy[3])
                if x2 > x1 and y2 > y1:
                    heatmap[y1:y2, x1:x2] = np.maximum(
                        heatmap[y1:y2, x1:x2], det.get('conf_std', 0))
            
            im = ax2.imshow(heatmap, alpha=0.6, cmap='jet', vmin=0, vmax=0.1)
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label='Uncertainty')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'visualization_{name}')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_report(self, detections, uncertainties, output_dir, name,
                     analysis_report=None, total_samples=None):
        path = os.path.join(output_dir, f'report_{name}.txt')
        sample_total = total_samples if total_samples else self.n_samples
        
        with open(path, 'w', encoding='utf-8') as f:
            if analysis_report:
                f.write(analysis_report + "\n\n")
            f.write("="*60 + "\n")
            f.write(f"检测报告 - {name}\n")
            f.write("="*60 + "\n\n")
            f.write(f"检测总数: {len(detections)}\n")
            f.write(f"平均conf_std: {uncertainties.get('avg_conf_std', 0):.4f}\n\n")

            cluster_summaries = uncertainties.get('clusters', [])
            f.write("聚类信息:\n")
            if cluster_summaries:
                for c in cluster_summaries:
                    score, _ = self.analyzer.calculate_quality_score(c)
                    f.write(
                        f"  ID:{c.get('cluster_id','-')} cls:{c.get('cls',0)} "
                        f"conf:{c.get('conf',0):.4f} std:{c.get('conf_std',0):.4f} "
                        f"samples:{c.get('num_samples',0)}/{sample_total} "
                        f"score:{score:.1f}\n"
                    )
            else:
                f.write("  无聚类结果\n")
            f.write("\n")
            
            for idx, det in enumerate(detections):
                obb = det['obb']
                xyxy = obb_to_xyxy(obb)
                f.write(f"#{idx+1}:\n")
                f.write(f"  置信度: {det['conf']:.4f} ± {det.get('conf_std', 0):.4f}\n")
                f.write(f"  样本数: {det['num_samples']}/{sample_total}\n")
                f.write(f"  边界: ({xyxy[0]:.0f},{xyxy[1]:.0f})-({xyxy[2]:.0f},{xyxy[3]:.0f})\n")
                f.write(f"  四角点: {obb.tolist()}\n\n")
        


def main():
    # ===================== 配置 =====================
    MODEL_PATH = './results/11s_base_3/train/weights/best.pt'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_SAMPLES = 30
    DROPOUT_RATE = 0.15
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    IMAGE_PATH = './data_crop/images'
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    OUTPUT_DIR = './results_bayesian'
    PREDICT_DIR = './results_bayesian/predict'
    # ================================================
    
    input_path = Path(IMAGE_PATH)
    if input_path.is_dir():
        image_paths = [
            p for p in sorted(input_path.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if not image_paths:
            return
    elif input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            return
        image_paths = [input_path]
    else:
        return

    try:
        # 贝叶斯推理
        engine = BayesianYOLOInference(
            model_path=MODEL_PATH, device=DEVICE,
            n_samples=N_SAMPLES, dropout_rate=DROPOUT_RATE,
            conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD
        )

        if input_path.is_dir():
            total_images = len(image_paths)
            for idx, image_path in enumerate(image_paths, start=1):
                print(f"[{idx}/{total_images}] {image_path.name}")
                engine.inference(
                    str(image_path),
                    OUTPUT_DIR,
                    predict_dir=PREDICT_DIR
                )
        else:
            for image_path in image_paths:
                engine.inference(
                    str(image_path),
                    OUTPUT_DIR,
                    predict_dir=PREDICT_DIR
                )
        
    except Exception:
        raise


def bayesian_detection(imaege_path, output_dir):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(current_dir, 'model/follicle/best.pt') 
    MODEL_PATH = model_file

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_SAMPLES = 25
    DROPOUT_RATE = 0.15
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}

    IMAGE_PATH = imaege_path
    OUTPUT_DIR = output_dir
    PREDICT_DIR = output_dir + '/predict'
    
    print(f"\n配置: device={DEVICE}, n={N_SAMPLES}, dropout={DROPOUT_RATE}")
    
    input_path = Path(IMAGE_PATH)
    if input_path.is_dir():
        image_paths = [
            p for p in sorted(input_path.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if not image_paths:
            print(f"\n❌ 目录中没有支持的图片: {IMAGE_PATH}")
            return
    elif input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            print(f"\n❌ 不支持的图片格式: {IMAGE_PATH}")
            return
        image_paths = [input_path]
    else:
        print(f"\n❌ 图像不存在: {IMAGE_PATH}")
        return


    engine = BayesianYOLOInference(
        model_path=MODEL_PATH, device=DEVICE,
        n_samples=N_SAMPLES, dropout_rate=DROPOUT_RATE,
        conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD
    )
    
    for image_path in image_paths:
        detections, uncertainties, analysis = engine.inference(
            str(image_path),
            OUTPUT_DIR,
            predict_dir=PREDICT_DIR
        )
        if detections:
            total_samples = engine.analyzer.total_samples
            print(f"\n各检测: {image_path.name}")
            for i, d in enumerate(detections):
                print(f"  #{i+1}: conf={d['conf']:.3f}, std={d.get('conf_std',0):.4f}, "
                        f"samples={d['num_samples']}/{total_samples}")
        