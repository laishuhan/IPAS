import os
import time
import requests
import shutil  # Added: Needed for clear_folder function

WATCH_DIR = "./database_test/temp_file"     # Directory where end_XXX files will be created
IMG_DIR_1 = "./database_test/img"             # Image directory 1
IMG_DIR_2 = "/home/work/md_ocr_test2/error_img" # Image directory 2
END_FILE_DIR = WATCH_DIR                    # End flag files will also be placed here


def upload_and_submit(img_path: str):
    """
    Send a single image to backend:
    1) /diagnosis/classify-image  upload image + callbackUrl
    2) /diagnosis/submit-date     send taskId
    """
    try:
        with open(img_path, 'rb') as f:
            files = {'file': f}
            data = {'callbackUrl': 'http://localhost:8000/receive_callback'}

            print(f"Start sending image: {img_path}")

            r = requests.post(
                'http://localhost:9090/diagnosis/classify-image',
                files=files,
                data=data,
                timeout=10
            )
            r.raise_for_status()

            resp_json = r.json()
            task_id = resp_json.get('data', {}).get('taskId')

            if not task_id:
                print(f"  Warning: taskId not found in response: {resp_json}")
                return

            time.sleep(1)

            payload = {'taskId': task_id}
            r2 = requests.post(
                'http://localhost:9090/diagnosis/submit-date',
                data=payload,
                timeout=10
            )
            r2.raise_for_status()

            print(f"Finished sending: {img_path}\n")

    except requests.RequestException as e:
        print(f"Request error, failed to send {img_path}: {e}\n")
    except Exception as e:
        print(f"Unknown error occurred while sending {img_path}: {e}\n")


def get_image_list(img_dir: str):
    """Scan directory and return sorted list of image file paths."""
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".pdf"}

    if not os.path.isdir(img_dir):
        return []

    img_list = [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ]

    img_list.sort()
    return img_list


def run_tests(img_list, indices, end_file_dir=END_FILE_DIR):
    """
    img_list: list of image paths
    indices:  1-based index list of images to test
    end_file_dir: directory to generate end flag file
    """

    os.makedirs(end_file_dir, exist_ok=True)

    max_idx = len(img_list)
    valid_indices = []

    for i in indices:
        if 1 <= i <= max_idx:
            valid_indices.append(i)
        else:
            print(f"Index {i} is invalid, skipped (valid range: 1 ~ {max_idx})")

    if not valid_indices:
        print("No valid image indices found. Test aborted.\n")
        return

    print(f"{len(valid_indices)} images will be sent...\n")

    last_img_path = None

    for i in valid_indices:
        img_path = img_list[i - 1]
        last_img_path = img_path
        upload_and_submit(img_path)
        time.sleep(0.1)

    # -------------------------------
    # Generate end_XXX file (no extension)
    # -------------------------------
    if last_img_path:
        last_name = os.path.basename(last_img_path)
        base_name = os.path.splitext(last_name)[0]
        end_filename = f"end_{base_name}"

        end_file_path = os.path.join(end_file_dir, end_filename)

        #time.sleep(2)

        with open(end_file_path, "w", encoding="utf-8") as f:
            f.write("done\n")

        print(f"\nEnd flag file created: {end_file_path}\n")


def print_image_list(img_list):
    """Display the list of images with index numbers, filename and file size (MB)."""
    if not img_list:
        print("No images found in directory.\n")
        return

    print("Images found:\n")
    for idx, path in enumerate(img_list, start=1):
        try:
            filename = os.path.basename(path)
            size_bytes = os.path.getsize(path)
            size_mb = size_bytes / (1024 * 1024)
            print(f"{idx} {filename}  ({size_mb:.2f} MB)")
        except OSError:
            print(f"{idx} {filename}  (size unavailable)")
    print("\n")



def select_directory():
    """Prompt user to select a directory."""
    while True:
        print("\nSelect Image Directory:")
        print(f"1. {IMG_DIR_1}")
        print(f"2. {IMG_DIR_2}")
        choice = input("Enter choice (1/2): ").strip()
        
        if choice == "1":
            return IMG_DIR_1
        elif choice == "2":
            return IMG_DIR_2
        else:
            print("Invalid choice. Please enter 1 or 2.")

def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return

    for f in os.listdir(folder_path):
        p = os.path.join(folder_path, f)
        if os.path.isfile(p) or os.path.islink(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p)

    print(f"Folder cleared: {folder_path}")


def main():
    # Initial directory selection
    current_img_dir = select_directory()
    
    if not os.path.isdir(current_img_dir):
        print(f"Warning: Image directory not found: {current_img_dir}")
    else:
        print(f"Selected: {current_img_dir}")

    img_list = get_image_list(current_img_dir)
    print_image_list(img_list)

    # Main loop — return to menu after each test
    while True:
        print(f"Current Directory: {current_img_dir}")
        print("Select test mode:")
        print("1. Test all images")
        print("2. Test a range (enter a, b)")
        print("3. Test a single image (enter an index)")
        print("4. Rescan image directory (refresh list)")
        print("5. Switch image directory")
        print("6. Exit program")
        print("7. Clear current directory") # Added option

        mode = input("Enter mode number (1-7): ").strip()

        if mode == "1":
            img_list = get_image_list(current_img_dir)
            if not img_list:
                print("No images available. Please rescan or add images.\n")
                continue
            print_image_list(img_list)
            indices = list(range(1, len(img_list) + 1))
            run_tests(img_list, indices)

        elif mode == "2":
            img_list = get_image_list(current_img_dir)
            if not img_list:
                print("No images available. Please rescan or add images.\n")
                continue
            print_image_list(img_list)
            try:
                a = int(input("Enter start index a: ").strip())
                b = int(input("Enter end index b: ").strip())
                if a > b:
                    a, b = b, a
                indices = list(range(a, b + 1))
                run_tests(img_list, indices)
            except ValueError:
                print("Invalid input. Operation cancelled.\n")

        elif mode == "3":
            img_list = get_image_list(current_img_dir)
            if not img_list:
                print("No images available. Please rescan or add images.\n")
                continue
            print_image_list(img_list)
            try:
                n = int(input("Enter image index: ").strip())
                run_tests(img_list, [n])
            except ValueError:
                print("Invalid input. Operation cancelled.\n")

        elif mode == "4":
            img_list = get_image_list(current_img_dir)
            print_image_list(img_list)

        elif mode == "5":
            # Switch directory logic
            current_img_dir = select_directory()
            if not os.path.isdir(current_img_dir):
                print(f"Warning: Image directory not found: {current_img_dir}")
            img_list = get_image_list(current_img_dir)
            print_image_list(img_list)

        elif mode == "6":
            print("Sender program exited.")
            break

        elif mode == "7":
            # Clear current directory logic
            print(f"WARNING: You are about to DELETE ALL FILES in: {current_img_dir}")
            confirm = input("Are you sure? (y/n): ").strip().lower()
            if confirm == 'y':
                clear_folder(current_img_dir)
                img_list = get_image_list(current_img_dir) # Refresh (should be empty)
            else:
                print("Operation cancelled.\n")

        else:
            print("Invalid selection. Please try again.\n")


if __name__ == "__main__":
    clear_folder(WATCH_DIR)
    main()
