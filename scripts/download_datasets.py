"""
Download flower datasets from Kaggle using kagglehub.
This will fetch both datasets and print their local paths.
"""
import kagglehub


def main():
    # l3llff/flowers
    path1 = kagglehub.dataset_download("l3llff/flowers")
    print("Path to l3llff/flowers:", path1)

    # TensorFlow team flower_photos
    path2 = kagglehub.dataset_download("batoolabbas91/flower-photos-by-the-tensorflow-team")
    print("Path to tensorflow-team flower_photos:", path2)

    print("\nDownload complete.")


if __name__ == "__main__":
    main()
