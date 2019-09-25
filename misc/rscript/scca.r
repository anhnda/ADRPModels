library(MASS)
library("scca")
args = commandArgs(trailingOnly=TRUE)
print (args[1])

IFOLD= 0
NC = 10

if (length(args) >= 1){
  IFOLD = args[1]
}
if (length((args) >= 2)){
  NC = args[2]
}

print (IFOLD)
print (NC)
library(matlib)


inputTrainP = paste0("/home/anhnd/DTI Project/Codes/MethodsComparisonX/data/RSCCA/trainInpAll_",IFOLD)
outputTrainP  = paste0("/home/anhnd/DTI Project/Codes/MethodsComparisonX/data/RSCCA/trainOutAll_",IFOLD)
#inputTest = paste0("/home/anhnd/DTI Project/Codes/BioDataLoader/out/data/inputTestP_",IFOLD)
#outputTest = paste0("/home/anhnd/DTI Project/Codes/BioDataLoader/out/data/outputTestP_",IFOLD)

print (inputTrainP)
X = as.matrix(utils::read.delim(inputTrainP,sep=" "))
Y = as.matrix(utils::read.delim(outputTrainP,sep=" "))

#X2 = as.matrix(utils::read.delim(inputTest,sep=" "))
#Y2 = as.matrix(utils::read.delim(outputTest,sep=" "))

print (dim(X))

scca.out = scca(X, Y, nc = NC, center = FALSE)
WX = scca.out$A
WY = scca.out$B

write.matrix(WX,paste0("/home/anhnd/DTI Project/Codes/MethodsComparisonX/data/RSCCA/WeightChem_",IFOLD))
write.matrix(WY,paste0("/home/anhnd/DTI Project/Codes/MethodsComparisonX/data/RSCCA/WeightSE_",IFOLD))


print(dim(WX))
print(dim(WY))



