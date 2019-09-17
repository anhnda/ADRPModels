Data=Liu

for ff in 1 0
do
  for m in "KNN" "RF" "LR" "CCA" "NN" "MF"
    do
          echo $m $Data $ff
          python main.py -m $m -d $Data -f $ff
    done
done
