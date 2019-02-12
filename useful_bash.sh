#随机乱序文件内容
shuf file.txt | more    
# 文件内容去重复：
ls ./*.jpg | sort -u | uniq | wc -l
#查找符合条件的文件并copy
cat ../pre_100.txt | xargs mv -t /d/pre_100_test/
cat ../pre_100.txt | xargs rm -rf 
cat ../pre_100.txt | xargs cp -t /d/pre_100_test/
#同样适用用find
find ./ -iname 'AAH_*.png' | xargs cp -t ../renji20181228_checked/AAH/

