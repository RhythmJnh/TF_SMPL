total=`ls -lah data/img/*.png | wc -l`

# for ((idx=0; idx<1; idx++))
for ((idx=0; idx<total; idx++))
do
	echo $idx
	python fitPose2d.py $idx
done