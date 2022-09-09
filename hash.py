import hashlib                                   #导入hashlib模块

def hash(file_path,Bytes=1024):
    md5_1 = hashlib.md5()                        #创建一个md5算法对象
    with open(file_path,'rb') as f:              #打开一个文件，必须是'rb'模式打开
        while 1:
            data =f.read(Bytes)                  #由于是一个文件，每次只读取固定字节
            if data:                             #当读取内容不为空时对读取内容进行update
                md5_1.update(data)
            else:                                #当整个文件读完之后停止update
                break
    ret = md5_1.hexdigest()                      #获取这个文件的MD5值
    return ret