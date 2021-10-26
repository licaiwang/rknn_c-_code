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

static int rknn_GetTop(
    float *pfProb,
    float *pfMaxProb,
    uint32_t *pMaxClass,
    uint32_t outputCount,
    uint32_t topNum)
{
    uint32_t i, j;

#define MAX_TOP_NUM 20
    if (topNum > MAX_TOP_NUM)
        return 0;

    memset(pfMaxProb, 0, sizeof(float) * topNum);
    memset(pMaxClass, 0xff, sizeof(float) * topNum);

    for (j = 0; j < topNum; j++)
    {
        for (i = 0; i < outputCount; i++)
        {
            if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) ||
                (i == *(pMaxClass + 3)) || (i == *(pMaxClass + 4)))
            {
                continue;
            }

            if (pfProb[i] > *(pfMaxProb + j))
            {
                *(pfMaxProb + j) = pfProb[i];
                *(pMaxClass + j) = i;
            }
        }
    }

    return 1;
}

rknn_input_output_num print_model_info(rknn_context ctx)
{
    int ret;
    printf("Reading Model INFO...");
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        throw std::runtime_error("rknn_query fail! ret");
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
    printf("input tensors:\n");
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
        printRKNNTensor(&(input_attrs[i]));
    }
    printf("output tensors:\n");
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
        printRKNNTensor(&(output_attrs[i]));
    }
    return io_num;
}

void predict_one_pic(cv::Mat img, rknn_context ctx, rknn_input_output_num io_num)
{
    int ret;
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
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
        uint32_t MaxClass[10];
        float fMaxProb[10];
        float *buffer = (float *)outputs[i].buf;
        uint32_t sz = outputs[i].size / 4;
        rknn_GetTop(buffer, fMaxProb, MaxClass, sz, 10);
        float max_prob = fMaxProb[0];
        for (int i = 0; i < 10; i++)
        {
            printf("%3d: %8.6f\n", MaxClass[i], fMaxProb[i]);
        }

        if (max_prob < 0.5)
        {
            printf("Predict Eyes Close \n\n");
        }
        else
        {
            printf("Predict Eyes Open\n\n");
        }
    }
    // Release rknn_outputs
    rknn_outputs_release(ctx, 1, outputs);
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
    const int MODEL_IN_WIDTH = 128;
    const int MODEL_IN_HEIGHT = 128;
    const int MODEL_IN_CHANNELS = 3;
    // Temp
    string path = "img/0_1.png";
    cout << "Read close eye pic from " << path << endl;
    // Load Temp image
    cv::Mat org_img = imread(path, cv::IMREAD_COLOR);
    input_checker(org_img);
    // Preprocess
    cv::Mat img = org_img.clone();
    cv::resize(img, org_img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);
    cv::cvtColor(img, org_img, COLOR_BGR2RGB);
    input_checker(img);
    // Check Input Type
    string ty = type2str(img.type());
    printf("Matrix: %s %dx%d \n", ty.c_str(), img.cols, img.rows);

    // Generate Test Data
    //    close index 到 2116
    //     open index 到 2117 ~ 6834
    generate_test_data(Close_eyes_path, 0);
    generate_test_data(Open_eyes_path, 1);

    // Load & Init RKNN Model And Caculate Time
    auto start = high_resolution_clock::now();
    model = load_model(model_path, &model_len);
    ret = rknn_init(&ctx, model, model_len, 0);
    auto stop = high_resolution_clock::now();
    if (ret < 0)
    {
        throw std::runtime_error("rknn load model fail!");
    }
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Load Model: " << duration.count() * micro << " seconds" << endl;

    // Get Model Input Output Info
    rknn_input_output_num io_num = print_model_info(ctx);

    // Predict One Picture
    predict_one_pic(img, ctx, io_num);

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