import cv2
from os import listdir, path
import tqdm
import src.detect_lines as dl


ASSETS_PATH = path.join(path.dirname(__file__), "assets")
# Camera paths L = left, R = right, G = gray, C = color
CAM_DATA = {
    "LG": path.join(ASSETS_PATH, "image_00", "data"),
    "RG": path.join(ASSETS_PATH, "image_01", "data"),
    "LC": path.join(ASSETS_PATH, "image_02", "data"),
    "RC": path.join(ASSETS_PATH, "image_03", "data"),
}


def write_video(images, file_name):
    output = path.join(ASSETS_PATH, "output")
    output_file = path.join(output, file_name)

    x_img, y_img, _ = images[0].shape

    out = cv2.VideoWriter(
        output_file, cv2.VideoWriter_fourcc(*"mp4v"), 15, (y_img, x_img)
    )

    for img in tqdm.tqdm(images):
        out.write(img)
    out.release()


def show_video(img):
    cv2.imshow("Edge detection", img)
    # wait for 10ms or till 'q' is pressed
    if cv2.waitKey(15) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        exit(0)


def main():
    images = listdir(CAM_DATA["LG"])
    images = [path.join(CAM_DATA["LG"], img) for img in images]

    out_img = []
    for img_path in tqdm.tqdm(images):
        img = cv2.imread(img_path)

        lines = dl.get_lines(img)

        # lines = dl.get_hough_p(img)
        img = dl.pre_processing(img)
        # dl.draw_lines(img, lines)

        show_video(img)
        # out_img.append(img)

    # write_video(out_img, "image_00_hp.mp4")


if __name__ == "__main__":
    main()
