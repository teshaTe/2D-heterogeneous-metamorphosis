#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <chrono>
#include <ctime>
#include <ratio>

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
  int n_frames     = 100;
  int move_img2_fr = 0;
  int block_size   = 1;
}

void print_args()
{
  std::cout << "Draw mode by default   - white"    << std::endl;
  std::cout << "Colour mode by default - rgb"      << std::endl;
  std::cout << "Block_size = " << args::block_size << std::endl;
  std::cout << "Frame_num  = " << args::n_frames   << std::endl;
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
  std::cout << "-b   - specify the texture block size \n";
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

    else if (!strcmp(cur, "-b"))
      args::block_size = std::stoi(argv[i + 1]);

    else if (!strcmp(cur, "-n"))
      args::n_frames = std::stoi(argv[i + 1]);

    else if (!strcmp(cur, "--back"))
      args::background = argv[i + 1];

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

    for(int y = 0; y < src.cols; ++y)
    {
        for(int x = 0; x < src.rows; ++x)
        {
            if(src_buf[x*src.channels()+3+y*src.step] == 0)
            {
                dst_buf[x*dst->channels()  +y*dst->step] = back_img_buf[x*back_img.channels()  +y*back_img.step];
                dst_buf[x*dst->channels()+1+y*dst->step] = back_img_buf[x*back_img.channels()+1+y*back_img.step];
                dst_buf[x*dst->channels()+2+y*dst->step] = back_img_buf[x*back_img.channels()+2+y*back_img.step];
                dst_buf[x*dst->channels()+3+y*dst->step] = back_img_buf[x*back_img.channels()+3+y*back_img.step];
            }
            else
            {
                dst_buf[x*dst->channels()  +y*dst->step] = src_buf[x*src.channels()  +y*src.step];
                dst_buf[x*dst->channels()+1+y*dst->step] = src_buf[x*src.channels()+1+y*src.step];
                dst_buf[x*dst->channels()+2+y*dst->step] = src_buf[x*src.channels()+2+y*src.step];
                dst_buf[x*dst->channels()+3+y*dst->step] = src_buf[x*src.channels()+3+y*src.step];
            }
        }
    }
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

    shader::ColorShader new_shader(args::n_frames, args::block_size, img1, img2, args::color_mode, args::draw_mode);

    cv::Mat dst = cv::Mat(img2.rows, img2.cols, img2.type());
    unsigned char *dst_buff = static_cast<unsigned char*>(dst.data);
    int frame_num = args::frame;

    std::cout << std::endl;

#ifdef USE_AFFINE_TRANSFORMATIONS
    std::vector<float> rec1_c = new_shader.get_rec1_center();
    std::vector<float> rec2_c = new_shader.get_rec2_center();

    float shift_x = (rec2_c[0] - rec1_c[0]) * img2.cols;
    float shift_y = (rec2_c[1] - rec1_c[1]) * img2.rows;

    std::vector<float> f2 = new_shader.get_f2_values();
#endif //USE_AFFINE_TRANSFORMATIONS

    auto start1 = std::chrono::high_resolution_clock::now();
    ngl::Vec4 col = ngl::Vec4(0.0, 0.0, 0.0, 0.0);

    while(frame_num < args::n_frames)
    {
       auto start2 = std::chrono::high_resolution_clock::now();

/*#ifdef USE_OPENMP
#pragma omp target teams distribute parallel for collapse(2) map(from: dst_buff) map(to: col, frame_num, img2, dst)
#endif*/
       for(int y = 0; y < img2.rows; y++ )
       {
           for(int x = 0; x < img2.cols; x++ )
           {
               new_shader.color_image( &col, ngl::Vec2(x, y), frame_num );
               dst_buff[x*dst.channels()  +y*dst.step] = col.m_b * 255;
               dst_buff[x*dst.channels()+1+y*dst.step] = col.m_g * 255;
               dst_buff[x*dst.channels()+2+y*dst.step] = col.m_r * 255;
               dst_buff[x*dst.channels()+3+y*dst.step] = col.m_a * 255;
           }
       }


       cv::Mat new_dst = cv::Mat(dst.cols,dst.rows,dst.type(), dst_buff);

       if(!args::background.empty())
       {
           cv::Mat final_dst = cv::Mat(new_dst.cols, new_dst.rows, new_dst.type());
           cv::Mat background = cv::imread(args::background, cv::IMREAD_UNCHANGED);

           if(background.empty())
           {
               std::cerr << "ERROR: failed to load background image [--back]! " << std::endl;
               return -1;
           }

           set_image_background(&final_dst, new_dst, background);
           std::string save_path = args::file_out + std::to_string(frame_num) + ".png";
           cv::imwrite(save_path, final_dst);
       }
       else
       {
           std::string save_path = args::file_out + std::to_string(frame_num) + ".png";
           cv::imwrite(save_path, new_dst);
       }

       auto end2 = std::chrono::high_resolution_clock::now();
       auto dur2 = std::chrono::duration_cast<std::chrono::duration<double>>(end2 - start2);
       std::cout << "image_"       <<frame_num     << " is saved." << std::endl;
       std::cout << "spent time: " << dur2.count() << " seconds"   << std::endl;

#ifdef USE_AFFINE_TRANSFORMATIONS
       auto start3 = std::chrono::high_resolution_clock::now();

       float st_x = 0.0f; float st_y = 0.0f;
       stb_tricks::StbEnhancements stb;
       stb.get_right_step_xy(&st_x, &st_y, shift_x, shift_y, 70);

       cv::Mat affine_m_st, affine_m;
       cv::Mat image2     = new_shader.get_image2();
       cv::Mat image2_sh  = stb.get_affine_transformed_image(image2, &affine_m_st, st_x, st_y);

       new_shader.update_affine_matrix(affine_m_st);
       affine_m = new_shader.get_affine_matrix();

       std::vector<float> f2_m;
       float res = 0.0f;

       for(int y = 0; y < image2_sh.rows; ++y)
       {
           for(int x = 0; x < image2_sh.cols; ++x)
           {
               float u_x = static_cast<float>(x) / static_cast<float>(image2_sh.rows);
               float u_y = static_cast<float>(y) / static_cast<float>(image2_sh.cols);
               new_shader.DF.get()->inverse_mapping_result(&res, f2, u_x, u_y, affine_m);
               f2_m.push_back(res);
           }
       }

       new_shader.update_image2(image2_sh);
       new_shader.update_im2_buff();
       new_shader.update_f2_tr_values(f2_m);
       new_shader.update_ai_coeffs();

       auto end3 = std::chrono::high_resolution_clock::now();
       auto dur3 = std::chrono::duration_cast<std::chrono::duration<double>>(end3 - start3);
       std::cout << "spent time(affine_op): " << dur3.count() << " seconds"   << std::endl;

       f2_m.clear();
#endif //USE_AFFINE_TRANSFORMATIONS

       frame_num += 1;
    }

    auto end1 = std::chrono::high_resolution_clock::now();
    auto dur1 = std::chrono::duration_cast<std::chrono::duration<double>>(end1 - start1);
    std::cout << "total time: " << dur1.count() << " seconds" << std::endl;

  return 0;
}
