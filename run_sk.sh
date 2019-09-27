Data=AEOLUS
# 
for ff in 2
do
  for m in "LNSM" "KNN"
    do
          echo $m $Data $ff
          python main.py -m $m -d $Data -f $ff
    done
done
