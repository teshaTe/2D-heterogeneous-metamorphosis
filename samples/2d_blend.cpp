#ifdef USE_OPENMP
  #include <omp.h>
#endif

#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <ratio>
#include <vector>

#include "include/heter2dSTB.h"
#include "include/timer.hpp"

hstb::DrawMode getDrawMode(std::string str)
{
    hstb::DrawMode d = hstb::DrawMode::NO_COLOR;
    if(str == "no_color")
        d = hstb::DrawMode::NO_COLOR;
    else if(str == "blend")
        d = hstb::DrawMode::COLOR_BLEND;
    else if(str == "closest")
        d = hstb::DrawMode::COLOR_CLOSEST;
    else
    {
        std::cerr << "ERROR: unknown draw mode specified! shoulld be [no_color, blend, closest]" << std::endl;
        exit(-1);
    }
    return d;
}

hstb::ColorMode getColorMode(std::string str)
{
    hstb::ColorMode c = hstb::ColorMode::RGB;
    if(str == "rgb")
        c = hstb::ColorMode::RGB;
    else if(str == "hsv")
        c = hstb::ColorMode::HSV;
    else if(str == "lab")
        c = hstb::ColorMode::CIELab;
    else
    {
        std::cerr << "ERROR: unknown color space specified! shoulld be [rgb, hsv, lab]" << std::endl;
        exit(-1);
    }
    return c;
}

hstb::BoundingSolid getBoundingSolidType(std::string str)
{
    hstb::BoundingSolid bSolid = hstb::BoundingSolid::halfPlanes;
    if(str == "halfPlanes")
        bSolid = hstb::BoundingSolid::halfPlanes;
    else if(str == "trPyramid")
        bSolid = hstb::BoundingSolid::truncPyramid;
    else if(str == "trCone")
        bSolid = hstb::BoundingSolid::truncCone;
    else
    {
        std::cerr << "ERROR: unknown bounding solid specified! shoulld be [halfPlanes, trPyramid, trCone]" << std::endl;
        exit(-1);
    }
    return bSolid;
}

namespace args
{
    std::string file_out;
    std::string start_pic;
    std::string final_pic;
    std::string background;

    hstb::DrawMode draw_mode   = hstb::DrawMode::NO_COLOR;
    hstb::ColorMode color_mode = hstb::ColorMode::RGB;
    hstb::BoundingSolid bSolid = hstb::BoundingSolid::halfPlanes;

    int frame        = 0;
    int n_frames     = 100;
    int move_img2_fr = 0;
    float scale      = 1.0f;
    bool useAffineTr = false;
}

void print_help()
{
    std::cout << "Default args are: \n" << std::endl;
    std::cout << "\nPossible customly specified options: \n";
    std::cout << "-im1     - specify input image \n";
    std::cout << "-im2     - specify target image \n";
    std::cout << "-d       - specify draw mode [white, blend, closest] \n";
    std::cout << "-c       - specify colour mode [rgb, hsv, lab] \n";
    std::cout << "--bsolid - specify bounding solid type [halfPlanes, trPyramid, trCone]\n";
    std::cout << "-s       - specify the scale for the picture [optional] \n";
    std::cout << "-n       - specify the number of output frames \n";
    std::cout << "           recomended: if bounding solid = 'truncated cone', n = 150+.\n\n";
    std::cout << "-o       - specify output path [optional] \n";
    std::cout << "--affine - set to enable affine transform to the target image 'im2; \n";
    std::cout << "--back   - specify image for background [optional] \n";
    std::cout << "--help   - output help\n" << std::endl;
}

void defaultParams()
{
    std::cout << "*****************************" << std::endl;
    std::cout << "Default Parameters:"         << std::endl;
    std::cout << "drawMode      = NO_COLOR;"   << std::endl;
    std::cout << "colorMode     = RGB;"        << std::endl;
    std::cout << "boundingSolid = halfPlanes;" << std::endl;
    std::cout << "total frames  = 100;"        << std::endl;
    std::cout << "image scaling = 1.0;"        << std::endl;
    std::cout << "use affine transform = false;" << std::endl;
    std::cout << "*****************************\n\n" << std::endl;
}

bool parse_args(int argc, char** argv)
{
    for (int i = 1; i < argc; i += 2)
    {
        const char *cur = argv[i];
        if(!strcmp(cur, "-im1"))
            args::start_pic = argv[i+1];

        else if(!strcmp(cur, "-im2"))
            args::final_pic = argv[i+1];

        else if (!strcmp(cur, "-o"))
            args::file_out = argv[i+1];

        else if (!strcmp(cur, "-d"))
            args::draw_mode = getDrawMode(argv[i+1]);

        else if (!strcmp(cur, "-c"))
            args::color_mode = getColorMode(argv[i+1]);

        else if (!strcmp(cur, "-s"))
            args::scale = std::stoi(argv[i+1]);

        else if (!strcmp(cur, "-n"))
            args::n_frames = std::stoi(argv[i+1]);

        else if(!strcmp(cur, "--affine"))
            args::useAffineTr = true;

        else if(!strcmp(cur, "--bsolid"))
            args::bSolid = getBoundingSolidType(argv[i+1]);

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
        std::cout << "\nWARNING: Output file is not specified, no data will be saved! \n" << std::endl;

    if( args::draw_mode == hstb::DrawMode::COLOR_BLEND || args::draw_mode == hstb::DrawMode::COLOR_CLOSEST )
        std::cout << "\nWARNING: expensive blending operation. Preview the output first using '-d no_color'\n" << std::endl;
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
        std::cerr << "\nERROR: failed to load img1 [-im1]! " << std::endl;
        return -1;
    }

    if(img2.empty())
    {
        std::cerr << "\nERROR: failed to load img2 [-im2]! " << std::endl;
        return -1;
    }

    defaultParams();

    hstb::heter2dSTB morphing(img1, img2, args::color_mode, args::draw_mode,
                                                args::n_frames, args::scale);

    if( !args::background.empty())
        morphing.setBackgroundImage( args::background );
    if( !args::file_out.empty())
        morphing.setOutputPath( args::file_out );

    morphing.computeMetamorphosis(args::bSolid, args::useAffineTr);

    return 0;
}


