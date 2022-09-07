from models import MyVGG
from utils import *
import json
from PIL import Image
import imageio
from torchvision.models import vgg19
from torchvision import transforms

if __name__ == "__main__":
    parser = initialize_parser()
    args = {arg: value for (arg, value) in parser.parse_args()._get_kwargs()}

    inference_image_path = args["inference-image-path"]
    vgg = vgg19(pretrained=False)
    myVgg = MyVGG(vgg)

    # Since we're using a model pre-trained
    # on ImageNet, it is recommended to
    # normalize the data with mean and std
    # obtained from the images available in
    # the ImageNet database
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    orig_img = Image.open(inference_image_path)
    img_arr = np.asarray(orig_img)
    transformed_img = transform(orig_img)
    transformed_img = transformed_img.view(
        (1, *transformed_img.shape)
    )  # shape: [height, width] --> [1, height, width]

    pred = myVgg(transformed_img)  # shape: [1, 1000]
    pred_class_index = torch.argmax(pred, dim=1).item()

    if args["verbose"]:
        with open(args["classes_index_path"], "r") as f:
            # You can find the JSON file from the following link:
            # https://github.com/raghakot/keras-vis/blob/master/resources/imagenet_class_index.json
            indexes_classes = json.load(f)

        pred_class = indexes_classes[str(pred_class_index)][1]
        print(f'The inference image is classified as "{pred_class}"!')

    if args["verbose"]:
        print("The weighted activations are being computed!")

    pred[0, pred_class_index].backward()
    gradients = myVgg.get_features_gradients()
    cnn_features = myVgg.get_cnn_features(transformed_img).detach()
    weighted_activations = get_weighted_activations(cnn_features, gradients)

    if args["verbose"]:
        print("The heatmaps are being produced!")

    all_combined_imgs = []

    for ch in range(weighted_activations.squeeze().shape[0]):
        heatmap = create_heatmap(
            weighted_activations,
            channel=ch,
            width=img_arr.shape[1],
            height=img_arr.shape[0],
        )
        combined_img = get_image_heatmap_combination(
            img_arr, heatmap, heatmap_ratio=args["heatmap_ratio"], channel=ch
        )
        all_combined_imgs.append(combined_img)

    avg_heatmap = create_heatmap(
        cnn_features, channel=None, width=img_arr.shape[1], height=img_arr.shape[0]
    )
    avg_combined_activations = get_image_heatmap_combination(
        img_arr, avg_heatmap, heatmap_ratio=args["heatmap_ratio"]
    )

    if args["save_output"]:
        if args["verbose"]:
            print("Saving the results...")

        cv2.imwrite(f"{args['output_name']}.jpg", avg_combined_activations)
        imageio.mimsave(f"{args['output_name']}.gif", all_combined_imgs, fps=8)
