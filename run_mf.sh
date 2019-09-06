Data=Aeolus
for m in "MF"
  do
        echo $m $Data
        python main.py -m $m -d $Data
	done

