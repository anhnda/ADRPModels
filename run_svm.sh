Data=AEOLUS
FF=2
for m in "SVM"
  do
        echo $m
        python main.py -m $m -d $Data -f $FF
	done

