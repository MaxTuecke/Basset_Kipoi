import numpy as np
import json, yaml
import types
import os, sys

from collections import OrderedDict

from os import listdir
from os.path import isfile, join

import kipoi.model
from kipoi.model import KerasModel, TensorFlowModel, SklearnModel, OldPyTorchModel, load_model_custom, PyTorchModel

from kipoi.data import PreloadedDataset, Dataset, BatchDataset, SampleIterator, SampleGenerator, BatchIterator, BatchGenerator
from kipoi.utils import load_module, cd, getargs
from kipoi.specs import DataLoaderDescription, ModelDescription

AVAILABLE_DATALOADERS = {"PreloadedDataset": PreloadedDataset,
                         "Dataset": Dataset,
                         "BatchDataset": BatchDataset,
                         "SampleIterator": SampleIterator,
                         "SampleGenerator": SampleGenerator,
                         "BatchIterator": BatchIterator,
                         "BatchGenerator": BatchGenerator}

DATALOADERS_AS_FUNCTIONS = ["PreloadedDataset", "SampleGenerator", "BatchGenerator"]


def run(inputParams, batchSize = None, stringInput = False, useGeneralLoader = True):

	inputParams = dict(inputParams)

	modelInfo = yaml.load(open("model/model.yaml"))
	loaderInfo = yaml.load(open("model/dataloader.yaml"))
	modelInfoJSON = json.load(open('./model/model_info.json'))
	MODEL_NAME = modelInfoJSON["model_name"]

	#TEMPORARY FOR TESTING
	#stringInput = "True"


	if stringInput == True:
		if not os.path.exists("./model/inputData"):
			os.makedirs("./model/inputData")
		write_files(inputParams, modelInfoJSON)

	model = createModel(modelInfo)
	dl_kwargs = makeDlKwargs(inputParams, modelInfoJSON, batchSize)

	if useGeneralLoader == True:
		try:
			from kipoi.specs import ModelDescription
			md = ModelDescription.load("./model/model_new.yaml")
			default_dataloader = md.default_dataloader.get()
			dl = default_dataloader(**dl_kwargs)
		except:
			dataLoader = getDataLoader()
			dl = dataLoader(**dl_kwargs)
	else:
		dataLoader = getDataLoader()
		dl = dataLoader(**dl_kwargs)

	if batchSize is not None:
		it = dl.batch_iter(batch_size=batchSize)
	else:
		it = dl.batch_iter(batch_size=modelInfoJSON["batch_size"])

	batch = next(it)

	if modelInfo["type"] != "custom":
		pred = model.predict_on_batch(batch['inputs'])
	else:
		try:
			modelObj = model(model_name = MODEL_NAME)
		except:
			modelObj = model()
		pred = modelObj.predict_on_batch(batch['inputs'])
	

	outputData = numpy_to_list(pred)
	outputData = check_targets(outputData, modelInfo)
	outputFormat = json.dumps(outputData)

	#delete_input_data()

	return outputFormat

def check_targets(outputData, modelInfo):
	try:
		targets_file = modelInfo["schema"]["targets"]["column_labels"][0]
		content = open("./model/"+targets_file).readlines()
		content = [x.strip() for x in content] 
		return {"Output Targets": outputData, "Column Labels": content}
	except:
		return {"Output Targets": outputData}
	


def numpy_to_list(input_list):
	if type(input_list) is np.ndarray:
		data = input_list.tolist()
	else:
		data = input_list
	try:
		for idx, item in enumerate(data): 
			data[idx] = numpy_to_list(item)
	except:
		pass
	return data

def write_files(inputParams, modelInfoJSON):
	fileNum = 1
	for key in inputParams:
		if modelInfoJSON["params_info"][key]["is_path"] == "True":
			with open("./model/inputData/file"+str(fileNum), 'wb') as f:
				f.write(inputParams[key])
			inputParams[key] = "file"+str(fileNum)
			fileNum = fileNum+1

def delete_input_data():
	import shutil
	folder = './model/inputData'
	for the_file in os.listdir(folder):
		if the_file != "input.txt":
			os.remove(os.path.join(folder, the_file))

def createModel(modelInfo):
	modelType = modelInfo["type"]
	modelArgs = modelInfo["args"]

	if modelType == "pytorch":
		model = createPytorchModel(modelInfo, modelArgs)
	elif modelType == "keras":
		model = createKerasModel(modelInfo, modelArgs)
	elif modelType == "custom": #CHECK THIS
		model = createCustomModel(modelInfo)
	elif modelType == "tensorflow":
		model = createTensorModel(modelInfo, modelArgs)
	elif modelType == "sklearn":
		model = createSklearnModel(modelInfo, modelArgs)
	else:
		raise ValueError('Error: Model type is not recogized')
	return model

def createPytorchModel(modelInfo, modelArgs):

	try:
		filePath = "./model/"+modelArgs["file"]
	except:
		filePath = None
	try:
		buildFN = modelArgs["build_fn"]
	except:
		buildFN = None
	try:
		weightsPath = "./model/"+modelArgs["weights"]
	except:
		weightsPath = None

	#module_name = os.path.basename(filePath)[:-3]
	#from importlib.machinery import SourceFileLoader
	#module = SourceFileLoader(module_name, filePath).load_module()
	#print(callable(module))

	#import importlib.util
	#import types
	#loader = importlib.machinery.SourceFileLoader(module_name, path)
	#module = types.ModuleType(loader.name)
	#loader.exec_module(module)

	#model = PyTorchModel(weights = "./model/model_files/pretrained_model_reloaded_th.pth", module_class = filePath, auto_use_cuda = False)

	model = OldPyTorchModel(file = filePath, build_fn = buildFN, weights = weightsPath, auto_use_cuda = False)

	return model

def createSklearnModel(modelInfo, modelArgs):
	modelPath = "./model/"+modelArgs["pkl_file"]
	try:
		predictMethod = modelArgs["predict_method"]
	except:
		predictMethod = "predict"

	model = SklearnModel(modelPath, predict_method = predictMethod)

	return model

def createKerasModel(modelInfo, modelArgs):
	weightsPath = "./model/"+modelArgs["weights"]
	try:
		archPath = "./model/"+modelArgs["arch"]
	except:
		archPath = None
	try:
		customPath = "./model/"+modelArgs["custom_objects"]
	except:
		customPath = None
	try:
		imagePath = modelArgs["image_dim_ordering"]
	except:
		imagePath = None


	model = KerasModel(weightsPath, arch=archPath, custom_objects = customPath, backend = None, image_dim_ordering = imagePath)
	return model

def createTensorModel(modelInfo, modelArgs):
	
	inputNodes = modelArgs["input_nodes"]
	targetNodes = modelArgs["target_nodes"]
	checkpointPath = "./model/"+modelArgs["checkpoint_path"]

	try:
		pklPath = "./model/"+modelArgs["const_feed_dict_pkl"]
	except:
		pklPath = None

	model = TensorFlowModel(inputNodes, targetNodes, checkpointPath, const_feed_dict_pkl = pklPath)

	return model

def createCustomModel(modelInfo):
	model = load_model_custom("./model/" + modelInfo["args"]["file"], modelInfo["args"]["object"])
	return model

def getDataLoader():

    yaml_path = './model/model.yaml'
    md = ModelDescription.load(yaml_path)

    if ":" in md.default_dataloader:
    	dl_path = md.default_dataloader.split(":")
    else:
    	dl_path = md.default_dataloader

    default_dataloader_path = './model/' + dl_path

    default_loader = get_dataloader_factory(default_dataloader_path)

    return default_loader

def get_dataloader_factory(dataloader):

    # pull the dataloader & get the dataloader directory
    yaml_path = './model/dataloader.yaml'
    dataloader_dir = './model/'

    # --------------------------------------------
    # Setup dataloader description
    with cd(dataloader_dir):  # move to the dataloader directory temporarily
        dl = DataLoaderDescription.load(os.path.basename(yaml_path))
        file_path, obj_name = tuple(dl.defined_as.split("::"))
        CustomDataLoader = getattr(load_module(file_path), obj_name)

    # check that dl.type is correct
    if dl.type not in AVAILABLE_DATALOADERS:
        raise ValueError("dataloader type: {0} is not in supported dataloaders:{1}".
                         format(dl.type, list(AVAILABLE_DATALOADERS.keys())))
    # check that the extractor arguments match yaml arguments
    if not getargs(CustomDataLoader) == set(dl.args.keys()):
        raise ValueError("DataLoader arguments: \n{0}\n don't match ".format(set(getargs(CustomDataLoader))) +
                         "the specification in the dataloader.yaml file:\n{0}".
                         format(set(dl.args.keys())))
    # check that CustomDataLoader indeed interits from the right DataLoader
    if dl.type in DATALOADERS_AS_FUNCTIONS:
        # transform the functions into objects
        assert isinstance(CustomDataLoader, types.FunctionType)
        CustomDataLoader = AVAILABLE_DATALOADERS[dl.type].from_fn(CustomDataLoader)
    else:
        if not issubclass(CustomDataLoader, AVAILABLE_DATALOADERS[dl.type]):
            raise ValueError("DataLoader does't inherit from the specified dataloader: {0}".
                             format(AVAILABLE_DATALOADERS[dl.type].__name__))

    # Inherit the attributes from dl
    CustomDataLoader.type = dl.type
    CustomDataLoader.defined_as = dl.defined_as
    CustomDataLoader.args = dl.args
    CustomDataLoader.info = dl.info
    CustomDataLoader.output_schema = dl.output_schema
    CustomDataLoader.dependencies = dl.dependencies
    CustomDataLoader.postprocessing = dl.postprocessing
    CustomDataLoader._yaml_path = yaml_path
    CustomDataLoader.source_dir = dataloader_dir
    #CustomDataLoader.print_args = classmethod(print_dl_kwargs)

    return CustomDataLoader

#this is messy clean up later
def makeDlKwargs(inputParams, modelInfoJSON, batchSize):
	#finds all the files that are needed

	"""
	neededDataListRaw = loaderInfo["args"]
	neededDataList = {}

	for key in neededDataListRaw:
		neededDataList[key] = "NAME"

	with open('./model/inputData/' + DATA_DEFINITIONS_FILE_NAME) as json_file:  
	    inputDataDirecotry = json.load(json_file)

	#replaces the spots in model info list that need file names that are associated with tags
	for key in inputDataDirecotry:
	    toChange = neededDataList[key]
	    if (((type(toChange).__name__) == "str" and sys.version_info[0] >= 3) or ((type(toChange).__name__) == "unicode" and sys.version_info[0] < 3)):
	        if(toChange == "NAME"):
	            neededDataList[key] = './model/inputData/' + inputDataDirecotry[key]
	    else:
	        import collections
	        for idx, val in enumerate(neededDataList[key]):
	            if val == "NAME":
	                neededDataList[key][idx] = './model/inputData/' + inputDataDirecotry[key]
	                break
	        neededDataList[key] = collections.OrderedDict([(neededDataList[key][0], neededDataList[key][1])])

	holder = []
	for key in neededDataList:
		if neededDataList[key] == "NAME":
			holder.append(key)
	for item in holder:
		neededDataList.pop(item)


	return neededDataList

    """
	modelInfoDataDict = modelInfoJSON["input_params"]
	paramInfo = modelInfoJSON["params_info"]
	outputDict = {}
	try:
		bs = loaderDataList["batch_size"]
		outputDict["batch_size"] = batchSize
	except:
		pass
	for key in inputParams:
		try:
			modelInfoDataDict[key]
		except:
			raise ValueError('Error: Unrecognized input param')

		if inputParams[key] is None and paramInfo[key]["is_optional"] == False:
			raise ValueError('Error: Missing input param: ' + key)
		if paramInfo[key]["is_path"] == False:
			outputDict[paramInfo[key]["param_name"]] = inputParams[key]
		else:
			outputDict[paramInfo[key]["param_name"]] = './model/inputData/' + inputParams[key]

	return outputDict


def test_run():
	#fasta_data = open("./model/TestFiles/hg38_chr22.fa", 'rb').read()
	#intervals_data = open("./model/TestFiles/intervals.bed", 'rb').read()
	#print(run({"intervals_file_name": intervals_data, "fasta_file_name": fasta_data}, batchSize=4, stringInput="True"))


	#print(run({"intervals_file_name": "intervals_file", "fasta_file_name": "fasta_file", "dnase_file_name": "dnase_file"}, batchSize=4, useGeneralLoader = "False"))

	#print(run({"anno_file_name": "SE_chr22.gtf", "fasta_file_name": "hg19_chr22.fa", "meth_file_name": "meth_chr22.bedGraph.sorted.gz"}, batchSize=4))
	#print(run({"vcf_file_name": "scn2a.vcf"}, batchSize=4))
	#print(run({"intervals_file_name": "intervals.bed", "fasta_file_name": "hg38_chr22.fa"}, batchSize=4))
	#print(run({"intervals_file_name": "intervals.bed", "fasta_file_name": "hg38_chr22.fa", "dnase_file_name": "dnase_synth.chr22.bw"}, batchSize=4))
	#print(run({"mirna_fasta_file_name": "miRNA.fasta", "mrna_fasta_file_name": "3UTR.fasta", "query_pair_file_name": "miRNA-mRNA_query.txt"}, batchSize=4))

	#print(run({"fasta_file_name": "hg38_chr22.fa", "intervals_file_name": "intervals.bed"}))
	#print(run(OrderedDict([('fasta_file_name', 'hg38_chr22.fa'), ('intervals_file_name', 'intervals.bed')]), batchSize = 4, useGeneralLoader = "False"))

#test_run()
