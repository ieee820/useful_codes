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
#使用！符号过滤掉不含有slice字样的图片
find ./ !  -iname '*slice*.png'


#mac os 
find . -name "plugin-*SNAPSHOT.jar" | xargs -J % cp -rp % /Users/user/Downloads/plugin-list/
ls | sort | head -30 | xargs -J % cp -rp % ../samples/3
