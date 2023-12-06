import math
import logging

import numpy as np
import matplotlib.image as mpimg
import open3d as o3d
from scipy.linalg import pinv


# DEMO kitti数据集位置
label_path="./label/000000.txt"
point_file = './velo/000000.bin'
calib_file="./calib/000000.txt"
seg_img = "./instance/000000.png"
out_path="./pointcloud.pcd"

# 初始化设置LOGGER
logging.basicConfig(level = logging.INFO,format='%(asctime)s|%(message)s')
# 创建
logger = logging.getLogger("img2velo")
logger.setLevel(logging.INFO)

# 创建handler
handler1=logging.FileHandler("./runtime.log")
handler1.setLevel(logging.INFO)
formatter=logging.Formatter('%(asctime)s|%(message)s')
handler1.setFormatter(formatter)

logger.addHandler(handler1)


def get_eu_dist(arr1, arr2):
    """计算两个点的欧式距离
    Arguments:
        arr1 {list} -- 1d list object with int or float
        arr2 {list} -- 1d list object with int or float
    Returns:
        float -- Euclidean distance
    """
    arr1=arr1[0:3]  # 剔除强度维度
    arr2=arr2[0:3]
    return sum((x1 - x2) ** 2 for x1, x2 in zip(arr1, arr2)) ** 0.5

def get_beta_angle(arr1, arr2):
    """计算两个点之间的夹角
    """
    arr1=arr1[0:3]
    arr2=arr2[0:3]
    OA = get_eu_dist(arr1, np.zeros(3))
    OB = get_eu_dist(arr2, np.zeros(3))
    AB = get_eu_dist(arr1, arr2)
    result = math.acos((OA**2+AB**2-OB**2)/(2*OA*OB))
    # logger.info(result)
    return result

def remove_ground(points):
    """去除地面点云"""
    no_ground = []
    for i in range(points.shape[1]):
        if(points[2][i]>-1.1):
            no_ground.append(i)
    points = points[:,no_ground]
    logger.info(f'去除地面点之后的点云shape:{points.shape}')
    return points


def load_point(all_para,file='./000011.bin'):
    """读取点云"""
    points = np.fromfile(file, dtype=np.float32).reshape([-1, 4]).T  # [4, n]
    logger.info(f'输入点云的形状为{points.shape}')
    # 将强度维度变为0
    points[3,:]=0
    # 去除地面点云
    points=remove_ground(points,all_para)
    logger.info(f'去除地面点之后的点云shape:{points.shape}')
    return points


def velo2cam(points,calib_file="000011_calib.txt"):
    """将雷达点云投影到相机平面"""
    # 读取标定信息
    calib_txt = []
    with open(calib_file, "r") as f:
        calib_txt = f.readlines()

    for i in range(len(calib_txt)):
        if calib_txt[i]!="\n":
            calib_txt[i] = calib_txt[i].split(":")[1]
    # 将点云投影到图片上
    # cam = P2 @ R0_rect @ Tr_velo_to_cam @ velo
    P2 = np.fromstring(calib_txt[2], dtype=np.float32, sep=" ").reshape((3,4))
    R0 = np.fromstring(calib_txt[4], dtype=np.float32, sep=" ").reshape((3,3))
    R0 = np.insert(R0,3,values=[0,0,0],axis=0)
    R0 = np.insert(R0,3,values=[0,0,0,1],axis=1)
    Tr_velo_to_cam = np.fromstring(calib_txt[5], dtype=np.float32, sep=" ").reshape((3,4))
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)

    cam = P2 @ R0 @ Tr_velo_to_cam @ points  # [3, n]
    cam[:2,:] /= cam[2,:]  # [x/z,y/z]
    return cam


def cam2velo(points,calib_file="000011_calib.txt"):
    """将相机平面点投影到点云平面（最终标注文件中的xyz坐标转换）
    输入图像点： [3, n]
    """
    # 读取标定信息
    calib_txt = []
    with open(calib_file, "r") as f:
        calib_txt = f.readlines()

    for i in range(len(calib_txt)):
        if calib_txt[i]!="\n":
            calib_txt[i] = calib_txt[i].split(":")[1]
    # 将点云投影到图片上
    # cam = P2 @ R0_rect @ Tr_velo_to_cam @ velo
    # velo = P2 @ R0_rect @ Tr_velo_to_cam / cam
    P2 = np.fromstring(calib_txt[2], dtype=np.float32, sep=" ").reshape((3,4))
    R0 = np.fromstring(calib_txt[4], dtype=np.float32, sep=" ").reshape((3,3))
    R0 = np.insert(R0,3,values=[0,0,0],axis=0)
    R0 = np.insert(R0,3,values=[0,0,0,1],axis=1)
    Tr_velo_to_cam = np.fromstring(calib_txt[5], dtype=np.float32, sep=" ").reshape((3,4))
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)
    velo = pinv(points)@P2 @ R0 @ Tr_velo_to_cam   # [3, n]
    return pinv(velo)


def imgseg2velo(points,cam,seg_img = "./instance_2/000011.png"):
    """依据图片分割结果为点云添加强度标注"""
    seg_png = mpimg.imread(seg_img)  # 读取分割标注

    a_set = {}  # 车的图片分割标记->强度标记
    b_set = {}  # 人
    a_begin = 50  # 车的强度标记起始点，之后每个单独个体增加5
    b_begin = 100  # 人

    points_out = []
    for i in range(cam.shape[1]):  # 遍历投影点云，寻找其是否有分割label，若存在则在添加强度标记
        y = round(cam[0][i])
        x = round(cam[1][i])
        if x>=seg_png.shape[0] or y>=seg_png.shape[1] or x<0 or y<0:
            continue
        if cam[0][i]>0 and cam[1][i]>0 and cam[2][i]>0:
            points_out.append(i)
        seg_label = seg_png[x][y]*1000
        if seg_label!=0:
            if seg_label>20:
                if seg_label in b_set:
                    points[3][i] = b_set[seg_label]
                else:
                    points[3][i] = b_begin
                    b_set[seg_label] = b_begin
                    b_begin+=5
            else:
                if seg_label in a_set:
                    points[3][i] = a_set[seg_label]
                else:
                    points[3][i] = a_begin
                    a_set[seg_label] = a_begin
                    a_begin+=5

    points = points.T  # [n, 4]

    # 过滤不在图片上的点
    points = points[points_out,:]
    logger.info(f'过滤不在图片上的点之后的点云shape为:{points.shape}')
    # 计算点云边界
    logger.info(f'点云范围：{get_point_limit(points)}')
    
    return points


def get_point_limit(points):
    """找到新的点云边界"""
    x_min, y_min, z_min = [float('inf'),float('inf'),float('inf')]
    x_max, y_max, z_max = [float('-inf'),float('-inf'),float('-inf')]
    
    for i in range(points.shape[0]):
        x_min = min(x_min,points[i][0])
        x_max = max(x_max,points[i][0])
        y_min = min(y_min,points[i][1])
        y_max = max(y_max,points[i][1])
        z_min = min(z_min,points[i][2])
        z_max = max(z_max,points[i][2])

    # logger.info(f'点云范围：{[x_min, x_max, y_min, y_max, z_min, z_max]}')
    return [x_min, x_max, y_min, y_max, z_min, z_max]


def export_pcd(points,out_path="./pointcloud2.pcd"):
    # 导出点云
    logger.info(points.shape)
    xyz = points[:,0:3]
    i = [[i] for i in points[:,3]]
    pcd = o3d.t.geometry.PointCloud()
    pcd.point["positions"] = o3d.core.Tensor(xyz)
    pcd.point["intensities"] = o3d.core.Tensor(i)
    o3d.t.io.write_point_cloud(out_path, pcd, write_ascii=True)
    logger.info(f"已导出点云pcd文件到：{out_path}")


def load_gt_labels(points, label_path="./label_2/000011.txt"):
    """读取label标签数据并转化为GT标注框"""
    pedes=[]
    car=[]
    cyclist=[]
    label_txt = []
    limits=get_point_limit(points)

    with open(label_path, "r") as f:
        label_txt = f.readlines()

    for i in range(len(label_txt)):
        label_txt[i] = label_txt[i].split()
        rect=[label_txt[i][13],label_txt[i][11],label_txt[i][12],label_txt[i][10],label_txt[i][9],label_txt[i][8],label_txt[i][14]]
        # rect=[label_txt[i][11],label_txt[i][12],label_txt[i][13],label_txt[i][10],label_txt[i][9],label_txt[i][8],label_txt[i][14]]
        rect=[float(i) for i in rect]
        # 从相机坐标系转换到雷达坐标系
        rect[0]=rect[0]+0.27
        rect[1]=-rect[1]  
        rect[2]=-rect[2]
        # rect[2]=rect[2]-0.08+rect[5]/2  # 标注的z轴坐标为三维框的底部中心坐标，需要转换到三维框的中心去
        rect[2]=rect[2]+0.08+rect[5]/2  # 标注的z轴坐标为三维框的底部中心坐标，需要转换到三维框的中心去

        # 判断该GT框是否在图像投影的点云边界内，如果不在则删除
        if rect[0]<limits[0] or rect[0]>limits[1] or rect[1]<limits[2] or rect[1]>limits[3]:
            # logger.info(f"去掉不在边界内的点")
            continue
        # 转换xyz坐标系
        if label_txt[i][0]=="Pedestrian":
            pedes.append(rect)
        elif label_txt[i][0]=="Car" or label_txt[i][0]=="Truck" or label_txt[i][0]=="Van":
            car.append(rect)
        elif label_txt[i][0]=="Cyclist ":
            cyclist.append(rect)
    return {'pedes':pedes, 'car':car, 'Cyclist':cyclist}


def visualize_box(points,bbox):
    """将所有在真实框内的点标记为50
       7个元素分别是中心点xyz坐标、箱子长宽高和偏航角（弧度制）
    """
    # 遍历所有点，在框中则改变强度
    for i in range(points.shape[0]):
        points[i][3]=0
        print(bbox)
        for k,label in bbox.items():
            width=max(label[3],label[4])/2
            center_x=label[0]
            center_y=label[1]
            if points[i][0]<center_x+width and points[i][0]>center_x-width and points[i][1]<center_y+width and points[i][1]>center_y-width and points[i][2]<label[2]+label[5]/2:
                points[i][3]=50
    export_pcd(points,"./检测物体框.pcd")


def print_cluster(clusters,points):
    """输出某个聚类簇团的点坐标"""
    for k,v in clusters.items():
        # logger.info(k)
        if k==50:
            logger.info(f'{k}:{np.around(points[v])}')
    logger.info(list(clusters.keys()))


def show_cluster(clusters,points):
    """在pcd点云文件中只显示某一种类的簇团"""
    points[:,3]=0
    cluster_point = clusters[50.0]
    for index in cluster_point:
        points[index][3]=100


def val_kitti_per_img(label_path,point_file, calib_file, seg_img, out_path, all_para):
    """读取kitti数据集中的单张图片，识别物体并计算IOU"""
    points = load_point(all_para, point_file)
    cam = velo2cam(points,calib_file)
    points = imgseg2velo(points,cam,seg_img)

    # 导出点云
    export_pcd(points)
    return points


if __name__=="__main__":
    # 验证单张图片
    points = val_kitti_per_img(label_path,point_file, calib_file, seg_img, out_path)
    # 导入真实标注
    # gt_labels = load_gt_labels(points)

