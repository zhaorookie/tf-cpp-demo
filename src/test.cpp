/*
#include <iostream>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

using namespace tensorflow;
using namespace tensorflow::ops;

int main()
{
  Scope root = Scope::NewRootScope();

  // Matrix A = [3 2; -1 0]
  auto A = Const(root, { {3.f, 2.f}, {-1.f, 0.f} });
  // Vector b = [3 5]
  auto b = Const(root, { {3.f, 5.f} });
  // v = Ab^T
  auto v = MatMul(root.WithOpName("v"), A, b, MatMul::TransposeB(true));

  std::vector<Tensor> outputs;
  ClientSession session(root);

  // Run and fetch v
  TF_CHECK_OK(session.Run({v}, &outputs));
  std::cout << "tensorflow session run ok" << std::endl;
  // Expect outputs[0] == [19; -3]
  std::cout << outputs[0].matrix<float>();

  return 0;
}
*/

// 下面注释的这段代码没有把mat转为tensor，所以会有损失
//#define COMPILER_MSVC
//#define NOMINMAX
//#define PLATFORM_WINDOWS   // 指定使用tensorflow/core/platform/windows/cpu_info.h


//#include<iostream>
//#include<opencv2/opencv.hpp>
//#include"tensorflow/core/public/session.h"
//#include "tensorflow/core/platform/env.h"
//#include <time.h>
//#include <vector>
//#include <string.h>
//using namespace tensorflow;
//using namespace cv;
//using std::cout;
//using std::endl;
//
//int main() {
//    const std::string model_path = "/home/ubuntu/learn/models/research/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb";// tensorflow模型文件，注意不能含有中文
//    const std::string image_path = "/home/ubuntu/learn/tensorflow-c++/objDet_export_tf2/image1.jpg";    // 待inference的图片grace_hopper.jpg
//
//    // 设置输入图像
//    cv::Mat img = cv::imread(image_path);
//    //cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
//    int height = img.rows;
//    int width = img.cols;
//    int depth = img.channels();
//
//    // 取图像数据，赋给tensorflow支持的Tensor变量中，创建一个tensor作为输入网络的接口
//    tensorflow::Tensor input_tensor(DT_UINT8, TensorShape({ 1, height, width, depth }));
//    const uint8* source_data = img.data;
//    auto input_tensor_mapped = input_tensor.tensor<uint8, 4>();
//
//    for (int i = 0; i < height; i++) {
//        const uint8* source_row = source_data + (i * width * depth);
//        for (int j = 0; j < width; j++) {
//            const uint8* source_pixel = source_row + (j * depth);
//            for (int c = 0; c < depth; c++) {
//                const uint8* source_value = source_pixel + c;
//                input_tensor_mapped(0, i, j, c) = *source_value;
//            }
//        }
//    }
//
//    /*--------------------------------创建session------------------------------*/
//    Session* session;
//    Status status = NewSession(SessionOptions(), &session);
//    if (!status.ok()) {
//        std::cerr << status.ToString() << endl;
//        return -1;
//    }
//    else {
//        cout << "Session created successfully" << endl;
//    }
//
//    /*--------------------------------从pb文件中读取模型--------------------------------*/
//    tensorflow::GraphDef graph_def;
//    status = ReadBinaryProto(Env::Default(), model_path, &graph_def);
//    if (!status.ok()) {
//        std::cerr << status.ToString() << endl;
//        return -1;
//    }
//    else {
//        cout << "Load graph protobuf successfully" << endl;
//    }
//
//    // 将graph加载到session
//    status = session->Create(graph_def);
//    if (!status.ok()) {
//        std::cerr << status.ToString() << endl;
//        return -1;
//    }
//    else {
//        cout << "Add graph to session successfully" << endl;
//    }
//
//    // 输入inputs，“ x_input”是我在模型中定义的输入数据名称
//    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
//            { "image_tensor:0", input_tensor },
//    };
//
//    // 输出outputs
//    std::vector<tensorflow::Tensor> outputs;
//
//    //批处理识别
//    double start = clock();
//    // num_detections等 通过 saved_model_cli show --dir saved_model --all查看后获得
//    std::vector<std::string> output_nodes;
//    output_nodes.push_back("num_detections");
//    output_nodes.push_back("detection_boxes");
//    output_nodes.push_back("detection_scores");
//    output_nodes.push_back("detection_classes");
//
//    /*-----------------------------------用网络进行测试-----------------------------------------*/
//    // 运行会话，最终结果保存在outputs中
//    status = session->Run(inputs, { output_nodes }, {}, &outputs);
//    if (!status.ok()) {
//        std::cerr << status.ToString() << endl;
//        return -1;
//    }
//    else {
//        cout << "Run session successfully" << endl;
//    }
//
//    std::vector<float> vecfldata;
//    std::vector<float> vecflprob;
//    // 接下来根据上面　output_nodes　的存放顺序（num_detections, detection_boxes, detection_scores, detection_classes）取值
//    for (int i = 0; i < outputs.size(); i++)
//    {
//        Tensor t = outputs[i]; // 从节点取出第一个输出 "node:0"
//        cout << "t.dtype(): " << t.dtype() << std::endl;
//        TensorShape shape = t.shape();
//        cout << "shape: " << shape << std::endl;
//        int dim = shape.dims();
//        cout << "dim: " << dim << std::endl;
//        cout << "shape.num_elements(): " << shape.num_elements() << std::endl;
//        std::vector<int> vecsize;
//        for (int d = 0; d < shape.dims(); d++)
//        {
//            int size = shape.dim_size(d);
//            cout << "size: " << size << endl;
//            vecsize.push_back(size);
//        }
//        if (dim == 3)  // dim等于３说明是框(detection_boxes), 因为框的shape是(1, 100, 4), 1是batchsize, 100是框的数量, ４是两个坐标点
//        {
//            auto tmap = t.tensor<float, 3>();//这里<float, 3>的3是根据dim=3来的
//            for (int l = 0; l < vecsize[0]; l++)
//            {
//                for (int m =0; m < vecsize[1]; m++)
//                {
//                    for (int n =0; n<vecsize[2];n++)
//                    {
//                        vecfldata.push_back(tmap(l,m,n));
//                    }
//                }
//            }
//
//        }
//        if (i==2)  // i == 2说明是detection_scores, 根据output_nodes可知
//        {
//            auto tmap = t.tensor<float, 2>();
//            for (int p = 0; p < shape.dim_size(1); p++)
//            {
//                vecflprob.push_back(tmap( 0, p));
//            }
//        }
//
//    }
//    for (int k = 0; k < vecfldata.size() / 4; k++)
//    {
//        if (vecflprob[k] > 0.7)
//        {
//            int lty = height*vecfldata[4 * k];
//            int ltx = width*vecfldata[4 * k + 1];
//            int rby = height*vecfldata[4 * k + 2];
//            int rbx = width*vecfldata[4 * k + 3];
//            cv::rectangle(img, cv::Point(ltx, lty), cv::Point(rbx, rby), Scalar(255, 0, 0));
//            cv::putText(img, std::to_string(vecflprob[k]), Point(ltx, lty), FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, Scalar(12, 255, 200), 1, 8);
//        }
//        else break;
//    }
//    std::cout << "outputs[0] num_detections" << outputs[0].DebugString() << std::endl;
//    std::cout << "outputs[1] detection_boxes" << outputs[1].DebugString() << std::endl;
//    std::cout << "outputs[2] detection_scores" << outputs[2].DebugString() << std::endl;
//    std::cout << "outputs[3] detection_classes" << outputs[3].DebugString() << std::endl;
//
//    double finish = clock();
//    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
//    cout << "spend time:" << duration << endl;
//    cv::imshow("image", img);
//    cv::waitKey();
//    return 0;
//}

#include <iostream>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <opencv2/opencv.hpp>
//#include <cv.h>
//#include <highgui.h>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace tensorflow;

// 定义一个函数讲OpenCV的Mat数据转化为tensor，python里面只要对cv2.read读进来的矩阵进行np.reshape之后，
// 数据类型就成了一个tensor，即tensor与矩阵一样，然后就可以输入到网络的入口了，但是C++版本，我们网络开放的入口
// 也需要将输入图片转化成一个tensor，所以如果用OpenCV读取图片的话，就是一个Mat，然后就要考虑怎么将Mat转化为
// Tensor了
void CVMat_to_Tensor(Mat img,Tensor* output_tensor,int input_rows,int input_cols)
{
    //imshow("input image",img);
    //图像进行resize处理
    resize(img,img,cv::Size(input_cols,input_rows));
    //imshow("resized image",img);

    //归一化
    img.convertTo(img,CV_8UC3);  // CV_32FC3
    //img=1-img/255;

    //创建一个指向tensor的内容的指针
    uint8 *p = output_tensor->flat<uint8>().data();

    //创建一个Mat，与tensor的指针绑定,改变这个Mat的值，就相当于改变tensor的值
    cv::Mat tempMat(input_rows, input_cols, CV_8UC3, p);
    img.convertTo(tempMat,CV_8UC3);

    //    waitKey(0);

}

int main()
{
    /*--------------------------------配置关键信息------------------------------*/
    string model_path="/home/ubuntu/learn/models/research/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb";
    string image_path="/home/ubuntu/learn/tensorflow-c++/objDet_export_tf2/image2.jpg";

    // 输入图片的宽和高
    int input_height = 900;
    int input_width = 1352;

    //　这里可以通过获取网络每一层的名字获得，下面有代码可以实现，也可以通过　saved_model_cli show --dir saved_model/ --all　获得
    string input_tensor_name="image_tensor";
    vector<string> out_put_nodes;  //注意，在object detection中输出的三个节点名称为以下三个
    out_put_nodes.push_back("detection_scores");  //detection_scores  detection_classes  detection_boxes
    out_put_nodes.push_back("detection_classes");
    out_put_nodes.push_back("detection_boxes");

    /*--------------------------------创建session------------------------------*/
    Session* session;
    Status status = NewSession(SessionOptions(), &session);//创建新会话Session

    /*--------------------------------从pb文件中读取模型--------------------------------*/
    GraphDef graphdef; //Graph Definition for current model

    Status status_load = ReadBinaryProto(Env::Default(), model_path, &graphdef); //从pb文件中读取图模型;
    if (!status_load.ok()) {
        cout << "ERROR: Loading model failed..." << model_path << std::endl;
        cout << status_load.ToString() << "\n";
        return -1;
    }

    // 获取网络每一层的名字
    cout << "names of network's layer: " << endl;
    for (int i=0; i < graphdef.node_size(); i++)
    {
        std::string name = graphdef.node(i).name();
        std::cout << name << std::endl;
    }

    Status status_create = session->Create(graphdef); //将模型导入会话Session中;
    if (!status_create.ok()) {
        cout << "ERROR: Creating graph in session failed..." << status_create.ToString() << std::endl;
        return -1;
    }
    cout << "<----Successfully created session and load graph.------->"<< endl;

    /*---------------------------------载入测试图片-------------------------------------*/
    cout<<endl<<"<------------loading test_image-------------->"<<endl;
    Mat img;
    img = imread(image_path);
    cvtColor(img, img, COLOR_BGR2RGB);
    if(img.empty())
    {
        cout<<"can't open the image!!!!!!!"<<endl;
        return -1;
    }

    //创建一个tensor作为输入网络的接口, 1表示batchsize, 3表通道　
    Tensor resized_tensor(DT_UINT8, TensorShape({1,input_height,input_width,3})); //DT_FLOAT

    //将Opencv的Mat格式的图片存入tensor
    CVMat_to_Tensor(img,&resized_tensor,input_height,input_width);

    cout << resized_tensor.DebugString()<<endl;

    /*-----------------------------------用网络进行测试-----------------------------------------*/
    cout<<endl<<"<-------------Running the model with test_image--------------->"<<endl;
    //前向运行，输出结果一定是一个tensor的vector
    vector<tensorflow::Tensor> outputs;

    Status status_run = session->Run({{input_tensor_name, resized_tensor}}, {out_put_nodes}, {}, &outputs);

    if (!status_run.ok()) {
        cout << "ERROR: RUN failed..."  << std::endl;
        cout << status_run.ToString() << "\n";
        return -1;
    }

    //把输出值给提取出
    cout << "Output tensor size:" << outputs.size() << std::endl;  //3
    for (int i = 0; i < outputs.size(); i++)
    {
        cout << outputs[i].DebugString()<<endl;   // [1, 50], [1, 50], [1, 50, 4]
    }

    cvtColor(img, img, COLOR_RGB2BGR);  // opencv读入的是BGR格式输入网络前转为RGB
    resize(img,img,cv::Size(1352,900));  // 模型输入图像大小
    int pre_num = outputs[0].dim_size(1);  // 50  模型预测的目标数量
    auto tmap_pro = outputs[0].tensor<float, 2>();  //第一个是score输出shape为[1,50]
    auto tmap_clas = outputs[1].tensor<float, 2>();  //第二个是class输出shape为[1,50]
    auto tmap_coor = outputs[2].tensor<float, 3>();  //第三个是coordinate输出shape为[1,50,4]
    float probability = 0.5;  //自己设定的score阈值
    for (int pre_i = 0; pre_i < pre_num; pre_i++)
    {
        if (tmap_pro(0, pre_i) < probability)
        {
            break;
        }
        cout << "Class ID: " << tmap_clas(0, pre_i) << endl;
        cout << "Probability: " << tmap_pro(0, pre_i) << endl;
        string id = to_string(int(tmap_clas(0, pre_i)));
        int xmin = int(tmap_coor(0, pre_i, 1) * input_width);
        int ymin = int(tmap_coor(0, pre_i, 0) * input_height);
        int xmax = int(tmap_coor(0, pre_i, 3) * input_width);
        int ymax = int(tmap_coor(0, pre_i, 2) * input_height);
        cout << "Xmin is: " << xmin << endl;
        cout << "Ymin is: " << ymin << endl;
        cout << "Xmax is: " << xmax << endl;
        cout << "Ymax is: " << ymax << endl;
        rectangle(img, Point(xmin, ymin), Point(xmax, ymax), Scalar(255, 0, 0), 1, 1, 0);
        putText(img, id, Point(xmin, ymin), FONT_HERSHEY_COMPLEX, 1.0, Scalar(255,0,0), 1);
    }
    imshow("1", img);
    waitKey(0);

//    session->Close();  //关闭Session

    return 0;
}