# xformer-paired-viz
Official repository for the upcoming WACV 2022 paper "Visualizing Paired Image Similarity in Transformer Networks". Here, you'll find the code to generate paired image visualizations using our provided ViT transformer models that were trained for the task of image retrieval.

# Instructions

Download the provided model weights hosted on Google drive [here](https://drive.google.com/drive/folders/1IJPFw6JsT9jtKHHeALcb4xNgAuRc5cqj?usp=sharing). For each of the 3 datasets that we used in our paper, Hotels-50k, Goolge LandmarksV2 (GLM), and Stanford Online Products (SOP), we provide the trained ViT-B16 model weights (ViT Base size with 16 patch-size) that were used to generate the visualizions in the paper. We also include the ResNet-101 weights for comparison. For Hotels-50k and SOP, we also include ViT-B32 weights.

After cloning the repo, make a new directory called "weights/{dataset}" and place the model weights there. 

The images to generate the similarity map for should go in the directory "images/{dataset}". We include some example images in this repo. To generate a paired-image visualization, run main.py using the command line arguments specifying the dataset, the filenames of the images (which should be stored in "images/{dataset}"), and the model type (ViT-B{16|32} or resnet-101) like so:

    python3 main.py --dataset Hotels-50k --imageA img1.png --imageB img2.png --model_type ViT-B16

The above command generates the paired similarity map visualizations of the images with the paths "images/Hotels-50k/img1.png" and "images/Hotels-50k/img2.png". The weights are loaded from "weights/Hotels-50k/ViT-B16.pth". Results are automatically saved in the directory "results/{dataset}/{model_type}", which for this example would be "results/Hotels-50k/ViT-B16". 

Image pair:

![image](https://user-images.githubusercontent.com/70965199/137340831-783d6fa6-23ad-431b-b695-301cf897b94a.png) ![image](https://user-images.githubusercontent.com/70965199/137340902-059ee951-538b-4abb-a9ab-f790c67bd60c.png)

Result:

![image](https://user-images.githubusercontent.com/70965199/137340994-bb40d94d-3a28-4ca4-9d0f-2e98adee870b.png) ![image](https://user-images.githubusercontent.com/70965199/137341020-d05e11c8-fc47-4ca5-8966-61897ba1d928.png)

# Citation

      @inproceedings{black2022visualizing,
      title={Visualizing Paired Image Similarity in Transformer Networks},
      author={Black, Samuel and Stylianou, Abby and Pless, Robert and Souvenir, Richard},
      booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
      pages={3164--3173},
      year={2022}}
