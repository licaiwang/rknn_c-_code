/****************************************************************************
*
*    Copyright (c) 2017 - 2018 by Rockchip Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of Rockchip Corporation. This is proprietary information owned by
*    Rockchip Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of Rockchip Corporation.
*
*****************************************************************************/

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <chrono>
#include <math.h>
#include <vector>
#include <cstring>
#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <stdexcept>
#include <sys/types.h>
#include <dirent.h>

using namespace std;
using namespace std::chrono;
using namespace cv;

/*-------------------------------------------
                  Globa Var
-------------------------------------------*/

const char *Close_eyes_path = "test/0";
const char *Open_eyes_path = "test/1";
vector<string> All_PiC_PATH;
vector<int> All_LABELS;

/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void input_checker(cv::Mat img)
{
    if (!img.data)
    {
        throw std::runtime_error("Error No Image");
    }
}

static void generate_test_data(const char *path, int label)
{
    DIR *directory_pointer;
    struct dirent *entry;
    if ((directory_pointer = opendir(path)) == NULL)
    {
        throw std::runtime_error("Error open Dir\n");
    }
    else
    {
        while ((entry = readdir(directory_pointer)) != NULL)
        {
            if (entry->d_name[0] == '.')
                continue;
            All_PiC_PATH.push_back(entry->d_name);
            All_LABELS.push_back(label);
        }
    }
}

static void printRKNNTensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
           attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char *)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model, 1, model_len, fp))
    {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    if (fp)
    {
        fclose(fp);
    }
    return model;
}

rknn_input_output_num print_model_info(rknn_context ctx)
{
    int ret;
    //printf("Reading Model INFO...");
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        throw std::runtime_error("rknn_query fail! ret");
    }
    //printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
    //printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            throw std::runtime_error("rknn_query fail! ret");
        }
        // printRKNNTensor(&(input_attrs[i]));
    }
    //printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            throw std::runtime_error("rknn_query fail! ret");
        }
        //printRKNNTensor(&(output_attrs[i]));
    }
    return io_num;
}

int predict_one_pic(cv::Mat img, rknn_context ctx, rknn_input_output_num io_num)
{
    int ret, predict;
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].size = img.cols * img.rows * img.channels();
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = img.data;
    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0)
    {
        throw std::runtime_error("rknn_inputs_set fail!");
    }
    ret = rknn_run(ctx, nullptr);
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0)
    {
        throw std::runtime_error("rknn_outputs_get fail!");
    }

    // Post Predict
    for (int i = 0; i < io_num.n_output; i++)
    {
        float *buffer = (float *)outputs[i].buf;
        if (buffer[0] > buffer[1])
        {
            predict = 0;
        }
        else
        {
            predict = 1;
        }
    }
    // Release rknn_outputs
    rknn_outputs_release(ctx, 1, outputs);
    return predict;
}

string type2str(int type)
{
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth)
    {
    case CV_8U:
        r = "8U";
        break;
    case CV_8S:
        r = "8S";
        break;
    case CV_16U:
        r = "16U";
        break;
    case CV_16S:
        r = "16S";
        break;
    case CV_32S:
        r = "32S";
        break;
    case CV_32F:
        r = "32F";
        break;
    case CV_64F:
        r = "64F";
        break;
    default:
        r = "User";
        break;
    }
    r += "C";
    r += (chans + '0');
    return r;
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{

    // Timer Staff
    const double micro = pow(10, -6);
    // RKNN Staff
    rknn_context ctx;
    unsigned char *model;
    const char *model_path = argv[1];
    const char *img_path = argv[2];
    const char *func = argv[3];
    int ret;
    int model_len = 0;
    // INPUT PIC INFO
    const int MODEL_IN_WIDTH = 96;
    const int MODEL_IN_HEIGHT = 96;
    const int MODEL_IN_CHANNELS = 1;

    // Generate Test Data
    //    close index 到 2116
    //     open index 到 2117 ~ 6834
    generate_test_data(Close_eyes_path, 0);
    generate_test_data(Open_eyes_path, 1);
    cout << "Generate test data done! " << endl;

    // Load & Init RKNN Model And Caculate Time
    // auto start = high_resolution_clock::now();
    model = load_model(model_path, &model_len);
    cout << "load model done! " << endl;
    ret = rknn_init(&ctx, model, model_len, 0);
    if (ret < 0)
    {
        throw std::runtime_error("rknn load model fail!");
    }
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
    // cout << "Load Model: " << duration.count() * micro << " seconds" << endl;
    // Get Model Input Output Info
    // rknn_input_output_num io_num = print_model_info(ctx);
    // Predict 1 Picture
    // predict_one_pic(img, ctx, io_num);
    // Confusion Matrix Staff
    double TP = 0.0;
    double FN = 0.0;
    double TN = 0.0;
    double FP = 0.0;
    // Run On Test data
    for (int i = 0; i < All_LABELS.size(); i++)
    {
        int tmp_res;
        int lab = All_LABELS[i];
        double minVal, maxVal;
        string img_path = All_PiC_PATH[i];
        string _path;

        if (lab == 0)
        {
            _path = "test/0/" + img_path;
        }

        if (lab == 1)
        {
            _path = "test/1/" + img_path;
        }

        // Load Temp image
        cv::Mat org_img = imread(_path, cv::IMREAD_COLOR);
        // Preprocess
        cv::Mat processed_img = cv::Mat::zeros(MODEL_IN_HEIGHT, MODEL_IN_WIDTH, CV_32F);
        cv::cvtColor(org_img, org_img, COLOR_BGR2GRAY);
        cv::resize(org_img, org_img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);
        minMaxLoc(org_img, &minVal, &maxVal);
        for (int i = 0; i != MODEL_IN_HEIGHT; ++i)
        {
            for (int j = 0; j != MODEL_IN_WIDTH; ++j)
            {
                float pixel = unsigned(org_img.at<uint8_t>(i, j));
                processed_img.at<float>(i, j) = 2 * ((pixel - minVal) / (maxVal - minVal)) - 1;
            }
        }
        cv::Mat img = processed_img.clone();
        rknn_input_output_num io_num = print_model_info(ctx);
        tmp_res = predict_one_pic(img, ctx, io_num);

        cout << "Read Image" << _path << endl;
        cout << "pic --- " << i + 1 << endl;

        if (tmp_res == lab && lab == 1)
        {
            TP += 1.0;
        }
        if (tmp_res != lab && lab == 1)
        {
            FN += 1.0;
        }

        if (tmp_res == lab && lab == 0)
        {
            TN += 1.0;
        }
        if (tmp_res != lab && lab == 0)
        {
            FP += 1.0;
        }
    }

    double ACC = (TP + TN) / (TP + FP + FN + TN);
    double PRECISION = TP / (TP + FP);
    double RECALL = TP / (TP + FN);
    double F1_SCORE = 2 / ((1 / PRECISION) + (1 / RECALL));

    cout << "TEST ACC ---" << ACC << endl;
    cout << "TEST PRECISION ---" << PRECISION << endl;
    cout << "TEST RECALL --- " << RECALL << endl;
    cout << "TEST F1_SCORE ---" << F1_SCORE << endl;
    cout << "TEST  FP ( Danger !!!)" << FP << endl;
    cout << "TEST  FN  ---" << FN << endl;

    // Release
    if (ctx >= 0)
    {
        rknn_destroy(ctx);
    }
    if (model)
    {
        free(model);
    }
    return 0;
}
