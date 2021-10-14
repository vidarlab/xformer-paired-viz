# xformer-paired-viz
Official repository for the upcoming WACV 2022 paper "Visualizing Paired Image Similarity in Transformer Networks". Here, you'll find the code to generate paired image visualizations using our provided ViT transformer models that were trained for the task of image retrieval.

# Instructions

Download the provided model weights hosted on Google drive [here](https://drive.google.com/drive/folders/1IJPFw6JsT9jtKHHeALcb4xNgAuRc5cqj?usp=sharing). For each of the 3 datasets that we used in our paper, Hotels-50k, Goolge LandmarksV2 (GLM), and Stanford Online Products (SOP), we provide the trained ViT-B/16 model weights that were used to generate the visualizions in the paper. We also include the ResNet-101 weights for comparison. 

After cloning the repo, make a new directory called "weights/" and place the model weights there. 

In the repo, we provide some example images under "examples/". To generate a visualization, run main.py using the command line arguments specifying the image pair, dataset, model weights, model type (ViT-B or resnet-101) and save directory, like so:

    python3 main.py --imageA examples/Hotels-50k/images/img1.png --imageB examples/Hotels-50k/images/img2.png --dataset Hotels-50k --model_weights weights/Hotels-50k/vit.pth --model_type ViT-B --save_dir examples/Hotels-50k/results
    
Image pair:

![image](https://user-images.githubusercontent.com/70965199/137340831-783d6fa6-23ad-431b-b695-301cf897b94a.png) ![image](https://user-images.githubusercontent.com/70965199/137340902-059ee951-538b-4abb-a9ab-f790c67bd60c.png)

Result:

![image](https://user-images.githubusercontent.com/70965199/137340994-bb40d94d-3a28-4ca4-9d0f-2e98adee870b.png) ![image](https://user-images.githubusercontent.com/70965199/137341020-d05e11c8-fc47-4ca5-8966-61897ba1d928.png)

# Citation

TODO: add citation once published.
