Data=Liu
for m in "KNN" "RF" "LR" "CCA" "NN"
  do
        echo $m
        python main.py -m $m -d $Data
	done

