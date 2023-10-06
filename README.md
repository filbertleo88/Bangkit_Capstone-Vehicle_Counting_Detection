# Introduction

The increase in population and transmigration from villages to cities, resulted in the growth of vehicle use, especially four-wheeled vehicles such as buses and trucks as a means of transportation commonly used by the community. The increasingly widespread development of toll roads also needs to be balanced with better traffic arrangements as well.

One of the problems often encountered in traveling using private and public vehicles through toll roads is the high density of roads, especially during the homecoming flow from villages to cities and vice versa during long holidays such as Eid and also the end and beginning of the year. To help solve these problems, the application of Intelligent Traffic System (ITS) on road sections can be a major consideration. Especially with the development of IoT and machine learning, ITS can also be further developed to solve more specific problems.

One of the branches of study in machine learning is computer vision, the application of computer vision can be done to detect and count vehicles passing on the highway. In vehicle detection, commonly used methods are background subtraction, frame difference, optical flow, and deep learning object detection. The first three methods detect vehicles through manually extracted features, which are relatively simple, but they also have some limitations in terms of accuracy or robustness. Instead of manually extracting features, deep learning methods simulate the information processing of the human brain and allow the built network to perform feature extraction automatically by training large annotated datasets. However, these methods rely on large training datasets and are difficult to apply to various traffic video scenarios. Transfer learning can be combined with deep learning to build a target task model based on the source task, but combining transfer learning with deep learning in the absence of annotated data is still an important research direction to study.

# Framework

![Framework](https://github.com/filbertleo88/Bangkit_Capstone-Vehicle_Counting_Detection/blob/main/image/Framework.png)

In developing the automatic vehicle counter model using computer vision, we divided the model framework into three phases as shown above. In the first phase, dataset building, we used annotated images with vehicles in the open dataset to avoid spending too much time labeling the data. In addition, there is an additional dataset that we extracted from a video that we took several frames in for additional model training and model testing and validation. The distribution of datasets for the model we developed is 65% training, 24% validation, and 11% testing from a total of 4000 images.

# Vehicle Detection Performance

The vehicle detection algorithm used is the YOLOV8n model developed by Ultralytic. Based on the benchmark results conducted for each YOLOV8 detection model, the following is a comparison of YOLOV8 performance compared to other versions of YOLO.

![Yolov8 Performance](https://github.com/filbertleo88/Bangkit_Capstone-Vehicle_Counting_Detection/blob/main/image/Yolov8%20Performance.png)

Model selection using YOLOV8n was carried out because its performance far exceeds the first alternative model choice, namely YOLOV5n. YOLOV8n selection also considers the mobility of alternative model choices. YOLOV8n is the YOLOV8 nano model or the smallest and lightest model of all YOLOV8 models.

![Model Training](https://github.com/filbertleo88/Bangkit_Capstone-Vehicle_Counting_Detection/blob/main/image/Model%20Training%20Epoch.png)

After evaluating the performance of the YOLOV8n model used in vehicle detection. The model produces a mAP50 value of 0.464 with 50 epochs. This value indicates a fairly good level of accuracy and speed in performing vehicle detection. The evaluation metric used is mAP50 (Mean Average Precision at 50) which combines precision and recall on the test dataset to get the average precision at 50% recall level. In this case, the mAP50 value of 0.464 indicates that the model is able to identify most vehicles with acceptable accuracy.

Although the mAP50 value of 0.464 does not reach a high level of accuracy, it is concluded that the value is acceptable in the context of using this model. The model is used specifically for vehicle detection and counting the number of vehicles passing a given point, so it does not require a very high level of mAP50 accuracy. The intended use of the model has been considered and the mAP50 value is considered adequate to achieve the stated objectives. In this context, possible detection errors are considered insignificant and tolerable in the vehicle count analysis.

However, it should be noted that the higher the mAP50 value, the more accurate and reliable the object detection produced by the model. If a higher level of accuracy is required or the model is used in more critical applications, efforts should be made to improve the performance of the model to achieve a higher mAP50 value. In future research, strategies such as adding more varied training data or adjusting the model architecture will be considered. In conclusion, the performance of the YOLOV8n model with a mAP50 value of 0.464 has provided acceptable results for the purpose of vehicle detection and number counting in the context described.

![Confusion Matrix Normalized](https://github.com/filbertleo88/Bangkit_Capstone-Vehicle_Counting_Detection/blob/main/image/confusion_matrix_normalized.png)
![Result](https://github.com/filbertleo88/Bangkit_Capstone-Vehicle_Counting_Detection/blob/main/image/results.png)

# Vehicle Counting

We use PyCharm or Visual Studio Code as a text editor in making this Python program. This program runs on a local computer machine where this program will run the vehicle detection process as well as the calculation of vehicles that pass certain limits, then the data from the number of vehicles entering and exiting and the capacity of the rest area will be stored in the form of a real-time database using Realtime Firebase, and also in the form of CSV in the form of a report of detected vehicles.

The Python program creation process begins by using the YOLO (You Only Look Once) model to detect objects in an image or video. The YOLO model is a deep learning algorithm that is efficient in detecting objects in real time. First, we import the necessary libraries, such as OpenCV for reading images or videos, and other libraries that support the implementation of the YOLO model.

After importing the necessary libraries, the next step is to load the YOLO model and its configuration. The model is pre-trained using training data that includes various objects to be detected. In this case, we will focus on detecting vehicles in rest areas. The YOLO model generates a bounding box and a label for each detected object.

After detecting the objects, the next step is to calculate the midpoint of each object that intersects with the predefined boundary line. This boundary line is usually drawn at the entrance or exit of the rest area to count vehicles entering or exiting. By using the object's midpoint coordinates and the equation of the line, we can determine whether the object is inside or outside the rest area.

Furthermore, the data from the number of vehicles counted will be processed to obtain information on incoming, and outgoing vehicles, and rest area capacity. The number of incoming vehicles will be calculated based on objects moving from outside to inside the limit line, while the number of outgoing vehicles is calculated based on objects moving from inside to outside the limit line. The rest area capacity can be calculated by comparing the number of incoming and outgoing vehicles with the maximum capacity that has been determined.

After processing the data, the last step is to save it into a CSV file and Realtime Database in Firebase. The CSV file serves as data storage that can be accessed and processed later. Meanwhile, the Realtime Database in Firebase allows us to store data in real time and access it from the Android mobile application. By using the Firebase API, we can send and retrieve data from the Android application to the database directly.

The data that has been saved into CSV and Realtime Database can then be processed again to create a graph that will be displayed in the Android mobile application. This graph will provide an easier-to-understand visualization of the number of incoming, outgoing, and incoming vehicles. 

![Python Program](https://github.com/filbertleo88/Bangkit_Capstone-Vehicle_Counting_Detection/blob/main/image/Python%20Program.png)
![Realtime Database Output](https://github.com/filbertleo88/Bangkit_Capstone-Vehicle_Counting_Detection/blob/main/image/Realtime%20Database.png)
![App Presentation](https://youtu.be/bOXsSVhAZwI?si=dAyheUGPNiLlNUXJ)
