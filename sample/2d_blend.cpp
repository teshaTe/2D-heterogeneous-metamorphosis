#ifdef USE_OPENMP
  #include <omp.h>
#endif

#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <chrono>
#include <ctime>
#include <ratio>
#include <vector>

#include "include/metamorphosis.h"

metamorphosis::DrawMode getDrawMode(std::string str)
{
    metamorphosis::DrawMode d = metamorphosis::DrawMode::WHITE;
    if(str == "white")
        d = metamorphosis::DrawMode::WHITE;

    else if(str == "blend")
        d = metamorphosis::DrawMode::SIMPLE_BLEND;

    else if(str == "closest")
        d = metamorphosis::DrawMode::SIMPLE_CLOSEST;

  return d;
}

metamorphosis::ColorMode getColorMode(std::string str)
{
    metamorphosis::ColorMode c = metamorphosis::ColorMode::RGB;
    if(str == "rgb")
        c = metamorphosis::ColorMode::RGB;

    else if(str == "hsv")
        c = metamorphosis::ColorMode::HSV;

    else if(str == "lab")
        c = metamorphosis::ColorMode::CIELab;

  return c;
}

namespace args
{
    std::string file_out;
    std::string start_pic;
    std::string final_pic;
    std::string background;

    metamorphosis::DrawMode draw_mode   = metamorphosis::DrawMode::WHITE;
    metamorphosis::ColorMode color_mode = metamorphosis::ColorMode::RGB;

    int frame        = 0;
#ifdef USE_CONE_F3
    int n_frames     = 150;
#else
    int n_frames     = 100;
#endif
    int move_img2_fr = 0;
    float scale      = 1.0f;
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
    std::cout << "-d   - specify draw mode [white, blend, closest] \n";
    std::cout << "-c   - specify colour mode [rgb, hsv, lab] \n";
    std::cout << "-s   - specify the scale for the picture [optional] \n";
    std::cout << "-n   - specify the number of output frames \n";
    std::cout << "-o   - specify output path [optional] \n";
    std::cout << "--back - specify image for background [optional] \n";
    std::cout << "--help - output help\n" << std::endl;
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
        else {
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

    if( args::file_out.empty() )
        std::cout << "\n WARNING: Output file is not specified, no data will be saved! \n" << std::endl;

    if( args::draw_mode == metamorphosis::DrawMode::SIMPLE_BLEND ||
        args::draw_mode == metamorphosis::DrawMode::SIMPLE_CLOSEST )
        std::cout << "WARNING: expensive blending operation. Preview the output first using '-d white'" << std::endl;

    std::cout << "Used arguments: \n";
    print_args();
    return true;
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

    metamorphosis::metamorphosis2D morphing( img1, img2, args::color_mode, args::draw_mode,
                                             args::n_frames, args::scale, true );

    //morphing.setManualSTBcoefficients( 1.5f, 1.0f, 1.0f, 1.0f );
    if( !args::background.empty())
        morphing.setBackgroundImage( args::background );
    if( !args::file_out.empty())
        morphing.setOutputPath( args::file_out );
    morphing.calculateMetamorphosis();

    return 0;
}


