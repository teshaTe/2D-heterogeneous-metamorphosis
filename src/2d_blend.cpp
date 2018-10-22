#ifdef USE_OPENMP
  #include <openmp/include/omp.h>
#endif

#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <chrono>
#include <ctime>
#include <ratio>
#include <vector>

#include "include/shader.h"

shader::DrawMode getDrawMode(std::string str)
{
  shader::DrawMode d = shader::DrawMode::WHITE;
  if(str == "white")
    d = shader::DrawMode::WHITE;

  else if(str == "block")
    d = shader::DrawMode::BLOCK_COLOR;

  else if(str == "blend")
    d = shader::DrawMode::SIMPLE_BLEND;

  else if(str == "closest")
    d = shader::DrawMode::SIMPLE_CLOSEST;

  return d;
}

shader::ColorMode getColorMode(std::string str)
{
  shader::ColorMode c = shader::ColorMode::RGB;
  if(str == "rgb")
    c = shader::ColorMode::RGB;

  else if(str == "hsv")
    c = shader::ColorMode::HSV;

  else if(str == "lab")
    c = shader::ColorMode::CIELab;

  return c;
}

namespace  args
{
  std::string file_out;
  std::string start_pic;
  std::string final_pic;
  std::string background;

  shader::DrawMode draw_mode   = shader::DrawMode::WHITE;
  shader::ColorMode color_mode = shader::ColorMode::RGB;

  int frame        = 0;
#ifdef USE_CONE_F3
  int n_frames     = 150;
#else
  int n_frames     = 100;
#endif
  int move_img2_fr = 0;
  float scale      = 0.0f;
}

void print_args()
{
  std::cout << "Draw mode by default   - white"  << std::endl;
  std::cout << "Colour mode by default - rgb"    << std::endl;
  std::cout << "Frame_num  = " << args::n_frames << std::endl;
  std::cout << "Scale = "      << args::scale    << std::endl;
}

void print_help()
{
  std::cout << "Default args are: \n" << std::endl;
  print_args();

  std::cout << "\nPossible customly specified options: \n";
  std::cout << "-im1 - specify first image \n";
  std::cout << "-im2 - specify final image \n";
  std::cout << "-d   - specify draw mode [white, block, blend, closest] \n";
  std::cout << "-c   - specify colour mode [rgb, hsv, lab] \n";
  std::cout << "-s   - specify the scale for the picture [optional] \n";
  std::cout << "-n   - specify the number of output frames \n";
  std::cout << "-o   - specify output path [optional] \n";
  std::cout << "--back - specify image for background [optional] \n";
  std::cout << "--help    - output help\n" << std::endl;
}

bool parse_args(int argc, char** argv)
{
  for (int i = 1; i < argc; i += 2)
  {
    const char *cur = argv[i];
    if(!strcmp(cur, "-im1"))
      args::start_pic = argv[i + 1];

    else if(!strcmp(cur, "-im2"))
      args::final_pic = argv[i + 1];

    else if (!strcmp(cur, "-o"))
      args::file_out = argv[i + 1];

    else if (!strcmp(cur, "-d"))
      args::draw_mode = getDrawMode(argv[i + 1]);

    else if (!strcmp(cur, "-c"))
      args::color_mode = getColorMode(argv[i + 1]);

    else if (!strcmp(cur, "-s"))
      args::scale = std::stoi(argv[i + 1]);

    else if (!strcmp(cur, "-n"))
      args::n_frames = std::stoi(argv[i + 1]);

    else if (!strcmp(cur, "--back"))
    {
      args::background = argv[i + 1];
      std::cout << "Your background is: " << args::background << std::endl;
    }

    else if (!strcmp(cur, "--help"))
    {
      print_help();
      return false;
    }
    else
    {
      std::cerr << "ERROR: Unknown Option! exiting..." << std::endl;
      return false;
    }
  }

  if(args::start_pic.empty())
  {
    std::cerr << "Please, specify the 1 image [-im1] for conducting blending operation!" << std::endl;
    return false;
  }
  else if (args::final_pic.empty())
  {
    std::cerr << "Please, specify the 2 image [-im2] for conducting blending operation!" << std::endl;
    return false;
  }

  if(args::file_out.empty())
    std::cout << "\n WARNING: Output file is not specified, no data will be saved! \n" << std::endl;

  if(args::draw_mode == shader::DrawMode::SIMPLE_BLEND || args::draw_mode == shader::DrawMode::SIMPLE_CLOSEST)
    std::cout << "WARNING: expensive blending operation. Preview the output first using '-d white'" << std::endl;

  std::cout << "Used arguments: \n";
  print_args();
  return true;
}

void set_image_background(cv::Mat *dst, cv::Mat src, cv::Mat back_img)
{
    unsigned char *dst_buf      = static_cast<unsigned char*>(dst->data);
    unsigned char *back_img_buf = static_cast<unsigned char*>(back_img.data);
    unsigned char *src_buf      = static_cast<unsigned char*>(src.data);
    int b_ch = back_img.channels();
    int d_ch = dst->channels();
    int s_ch = src.channels();
    size_t b_step = back_img.step;
    size_t d_step = dst->step;
    size_t s_step = src.step;
    int im_w = src.cols;
    int im_h = src.rows;

#ifdef USE_OPENMP
#pragma omp parallel shared(im_h, im_w, s_ch, s_step, d_ch, d_step, s_ch, s_step, b_ch, b_step, dst_buf, back_img_buf, src_buf)
{
#pragma omp for simd schedule(static,5)
#endif
    for(int y = 0; y < im_h; ++y)
    {
        for(int x = 0; x < im_w; ++x)
        {
            if(src_buf[x*s_ch+3+y*s_step] == 0)
            {
                dst_buf[x*d_ch  +y*d_step] = back_img_buf[x*b_ch  +y*b_step];
                dst_buf[x*d_ch+1+y*d_step] = back_img_buf[x*b_ch+1+y*b_step];
                dst_buf[x*d_ch+2+y*d_step] = back_img_buf[x*b_ch+2+y*b_step];
                dst_buf[x*d_ch+3+y*d_step] = back_img_buf[x*b_ch+3+y*b_step];
            }
            else
            {
                dst_buf[x*d_ch  +y*d_step] = src_buf[x*s_ch  +y*s_step];
                dst_buf[x*d_ch+1+y*d_step] = src_buf[x*s_ch+1+y*s_step];
                dst_buf[x*d_ch+2+y*d_step] = src_buf[x*s_ch+2+y*s_step];
                dst_buf[x*d_ch+3+y*d_step] = src_buf[x*s_ch+3+y*s_step];
            }
        }
    }
#ifdef USE_OPENMP
}
#endif
}


int main(int argc, char** argv)
{
    if (!parse_args(argc, argv))
      exit(EXIT_SUCCESS);

    cv::Mat img1 = cv::imread(args::start_pic, cv::IMREAD_UNCHANGED);
    cv::Mat img2 = cv::imread(args::final_pic, cv::IMREAD_UNCHANGED);

    if(img1.empty())
    {
        std::cerr << "ERROR: failed to load img1 [-im1]! " << std::endl;
        return -1;
    }

    if(img2.empty())
    {
        std::cerr << "ERROR: failed to load img2 [-im2]! " << std::endl;
        return -1;
    }

    cv::Mat img11, img22;

    if(args::scale > 0)
    {
        img11 = cv::Mat(img1.rows/args::scale, img1.cols/args::scale, img1.type());
        img22 = cv::Mat(img2.rows/args::scale, img2.cols/args::scale, img2.type());

        cv::resize(img1, img11, img11.size(), 0, 0,cv::INTER_NEAREST);
        cv::resize(img2, img22, img22.size(), 0, 0,cv::INTER_NEAREST);
    }
    else
    {
        img11 = cv::Mat(img1.rows, img1.cols, img1.type());
        img22 = cv::Mat(img2.rows, img2.cols, img2.type());
        img1.copyTo(img11);
        img2.copyTo(img22);
    }

    shader::ColorShader new_shader(args::n_frames, img11, img22, args::color_mode, args::draw_mode);

    cv::Mat dst = cv::Mat(img22.rows, img22.cols, img22.type());
    unsigned char *dst_buff = static_cast<unsigned char*>(dst.data);
    int frame_num = args::frame;

    std::cout << std::endl;
    int im_w = img22.cols;
    int im_h = img22.rows;

#ifdef USE_AFFINE_TRANSFORMATIONS
    std::vector<float> rec1_c = new_shader.get_rec1_center();
    std::vector<float> rec2_c = new_shader.get_rec2_center();

    float shift_x = (rec2_c[0] - rec1_c[0]) * im_w;
    float shift_y = (rec2_c[1] - rec1_c[1]) * im_h;

    std::vector<float> f2 = new_shader.get_f2_values();
    std::vector<float> affine_time;
#endif //USE_AFFINE_TRANSFORMATIONS

    auto start1 = std::chrono::high_resolution_clock::now();
    ngl::Vec4 col = ngl::Vec4(0.0, 0.0, 0.0, 0.0);

    std::vector<double> time;
    size_t d_step = dst.step;
    int d_ch = dst.channels();

#ifdef USE_CROSS_DISSOLVE
    int ch1 = img1.channels();
    int ch2 = img2.channels();
    size_t step1 = img1.step;
    size_t step2 = img2.step;
    unsigned char *im1_buf = img1.data;
    unsigned char *im2_buf = img2.data;
#endif

    while(frame_num < args::n_frames)
    {
       auto start2 = std::chrono::high_resolution_clock::now();

//calculating STB+STTI over each frame
       for(int y = 0; y < im_h; y++ )
       {
           for(int x = 0; x < im_w; x++ )
           {
#ifndef USE_CROSS_DISSOLVE
               new_shader.color_image( &col, ngl::Vec2(x,y), frame_num );
               dst_buff[x*d_ch  +y*d_step] = col.m_b * 255;
               dst_buff[x*d_ch+1+y*d_step] = col.m_g * 255;
               dst_buff[x*d_ch+2+y*d_step] = col.m_r * 255;
               dst_buff[x*d_ch+3+y*d_step] = col.m_a * 255;
#else
               float time = static_cast<float>(frame_num) / static_cast<float>(args::n_frames);

               dst_buff[x*d_ch  +y*d_step] = (1.0f-time)*im1_buf[x*ch1   + y*step1] + time * im2_buf[x*ch2   + y*step2];
               dst_buff[x*d_ch+1+y*d_step] = (1.0f-time)*im1_buf[x*ch1+1 + y*step1] + time * im2_buf[x*ch2+1 + y*step2];
               dst_buff[x*d_ch+2+y*d_step] = (1.0f-time)*im1_buf[x*ch1+2 + y*step1] + time * im2_buf[x*ch2+2 + y*step2];
               dst_buff[x*d_ch+3+y*d_step] = (1.0f-time)*im1_buf[x*ch1+3 + y*step1] + time * im2_buf[x*ch2+3 + y*step2];
#endif
           }
       }

       //writing image with or without background
       if(!args::background.empty())
       {
           cv::Mat background = cv::imread(args::background, cv::IMREAD_UNCHANGED);
           if(background.empty())
           {
               std::cerr << "ERROR: failed to load background image [--back]! " << std::endl;
               return -1;
           }

           if(args::scale > 0)
           {
               cv::Mat final_dst = cv::Mat(im_w*args::scale, im_h*args::scale, dst.type());
               cv::Mat dst0      = cv::Mat(im_w*args::scale, im_h*args::scale, dst.type());
               cv::resize(dst, dst0, dst0.size(), cv::INTER_AREA);

               set_image_background(&final_dst, dst0, background);
               std::string save_path = args::file_out + std::to_string(frame_num) + ".png";
               cv::imwrite(save_path, final_dst);
           }
           else
           {
               cv::Mat final_dst = cv::Mat(im_w, im_h, dst.type());

               set_image_background(&final_dst, dst, background);
               std::string save_path = args::file_out + std::to_string(frame_num) + ".png";
               cv::imwrite(save_path, final_dst);
           }
       }
       else
       {
           std::string save_path = args::file_out + std::to_string(frame_num) + ".png";
           cv::imwrite(save_path, dst);
       }

       auto end2 = std::chrono::high_resolution_clock::now();
       auto dur2 = std::chrono::duration_cast<std::chrono::duration<double>>(end2 - start2);

       std::cout << "image_"       <<frame_num     << " is saved." << std::endl;
       std::cout << "spent time: " << dur2.count() << " seconds"   << std::endl;
       time.push_back(dur2.count());

#ifdef USE_AFFINE_TRANSFORMATIONS
       auto start3 = std::chrono::high_resolution_clock::now();

       float st_x = 0.0f;
       float st_y = 0.0f;
       stb_tricks::StbEnhancements stb;
       stb.get_right_step_xy(&st_x, &st_y, shift_x, shift_y, 70);

       if(st_x != 0.0f && st_y != 0.0f)
       {
            cv::Mat affine_m_st, affine_m;
            cv::Mat image2     = new_shader.get_image2();
            cv::Mat image2_sh  = stb.get_affine_transformed_image(image2, &affine_m_st, st_x, st_y);

#ifdef USE_DEBUG_INFO
            cv::imwrite("tmp.png", image2_sh);
#endif
            new_shader.update_affine_matrix(affine_m_st);
            affine_m = new_shader.get_affine_matrix();

            std::vector<float> f22_m;
            f22_m.resize(f2.size());
            float *f2_m = f22_m.data();
            float u_x, u_y;
            float res = 0.0f;

            for(int y = 0; y < im_h; ++y)
            {
                for(int x = 0; x < im_w; ++x)
                {
                    u_x = static_cast<float>(x) / static_cast<float>(im_h);
                    u_y = static_cast<float>(y) / static_cast<float>(im_w);
                    new_shader.DF.get()->inverse_mapping_result(&res, f2, u_x, u_y, affine_m);
                    f2_m[x+y*im_w] = res;
                }
            }

            new_shader.update_image2(image2_sh);
            new_shader.update_im2_buff();
            new_shader.update_f2_tr_values(f22_m);
            new_shader.update_ai_coeffs();
            f22_m.clear();

            auto end3 = std::chrono::high_resolution_clock::now();
            auto dur3 = std::chrono::duration_cast<std::chrono::duration<double>>(end3 - start3);
            std::cout << "spent time(affine_op): " << dur3.count() << " seconds"   << std::endl;
            affine_time.push_back(dur3.count() + dur2.count());
       }

#endif //USE_AFFINE_TRANSFORMATIONS

       frame_num += 1;
    }

    auto end1 = std::chrono::high_resolution_clock::now();
    auto dur1 = std::chrono::duration_cast<std::chrono::duration<double>>(end1 - start1);
    std::cout << "total time: " << dur1.count() << " seconds" << std::endl;

#ifndef USE_AFFINE_TRANSFORMATIONS
    std::cout << "min_time per frame: "     << *std::minmax_element(time.begin(), time.end()).first  << std::endl;
    std::cout << "max_time per frame: "     << *std::minmax_element(time.begin(), time.end()).second << std::endl;
    std::cout << "average_time per frame: " << dur1.count() / args::n_frames <<std::endl;
#else
    std::cout << "min_time per frame: "     << *std::minmax_element(affine_time.begin(), affine_time.end()).first  << std::endl;
    std::cout << "max_time pre frame: "     << *std::minmax_element(affine_time.begin(), affine_time.end()).second << std::endl;
    std::cout << "average_time per frame: " << dur1.count() / args::n_frames << std::endl;
#endif

  return 0;
}
