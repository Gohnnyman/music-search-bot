from imageai.Classification import ImageClassification

prediction = ImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath("Imageprediction/resnet50-19c8e357.pth")
prediction.loadModel()

def predict(): 


	predictions, probabilities = prediction.classifyImage("Imageprediction/img.jpg", result_count = 3)
	for eachPrediction, eachProbability in zip(predictions, probabilities):
		print(eachPrediction , " : " , eachProbability)

	return predictions, probabilities





