Data=Liu
# 
for ff in 2
do
  for m in "KNN" "RF" "LR" "CCA" "MLN" "MF"
    do
          echo $m $Data $ff
          python main.py -m $m -d $Data -f $ff
    done
done
