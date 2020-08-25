#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "include/render2D.h"
#include <SFML/Graphics.hpp>

#include <algorithm>
#include <cmath>

namespace hfrep2D {

void render2D::createWindow(int resX, int resY, std::string title)
{
    window = new sf::RenderWindow(sf::VideoMode( resX, resY), title );
    window->setFramerateLimit(60);
}

void render2D::displayImage(sf::Image img)
{
    createWindow( img.getSize().x, img.getSize().y, "image viewer" );
    sf::Texture texture;
    texture.loadFromImage(img);
    sf::Sprite sprite;
    sprite.setTexture(texture, true);

    while( window->isOpen() )
    {
        sf::Event event;
        while( window->pollEvent(event) )
        {
            if( event.type == sf::Event::Closed )
            {
                window->close();
            }
            else if(event.type == sf::Event::KeyPressed )
            {
                if(event.key.code == sf::Keyboard::Escape)
                {
                    window->close();
                }
            }
        }
        window->clear(sf::Color::White);
        window->draw( sprite );
        window->display();
    }
}

void render2D::convertFieldToImage(std::vector<unsigned char> *uField, const std::vector<float> *field,
                                        int resX, int resY)
{
    uField->clear();
    uField->resize(resX*resY);
    float dist;

    for( int i = 0; i < resX*resY; i++ )
    {
        if( std::abs(field->at(i)) < 0.0009f )
            dist = std::abs( field->at(i) ) * 20.0f*1000.0f + 128.0f;
        else
            dist = std::abs( field->at(i) ) * 20.0f + 128.0f;

        if( dist < 0.0f )
            uField->at(i) = 0;
        else if( dist > 255.0f )
            uField->at(i) = 255;
        else
            uField->at(i) = static_cast<unsigned char>(dist);
    }
}

sf::Image render2D::drawIsolines(const std::vector<float> *field, int resX, int resY, float thres,
                            std::string fileName, float thresISO)
{
    std::vector<unsigned char> uField;
    convertFieldToImage( &uField, field, resX, resY );

    sf::Image dst; dst.create( resX, resY );
    int col_ind = 0;
    int col_st  = 5;

    for(int y = 0; y < resY; y++)
    {
        for(int x = 0; x < resX; x++)
        {
            for(int i = 0; i <= 255-col_st; i+=col_st)
            {
                if( field->at(x+y*resX) < 0.0f )
                {
                    float val = field->at(x+y*resX);
                    float indicator = std::fmod(val, static_cast<float>(1.0/col_st));

                    if( indicator < 0.0f && indicator > -thresISO )
                    {
                        dst.setPixel( x, y, sf::Color(100, 100, 100) );
                        break;
                    }

                    if( i <= uField[x+y*resX] && uField[x+y*resX] <= i+col_st )
                    {
                        dst.setPixel( x, y, sf::Color(0, i+col_st, i+col_st) );
                        break;
                    }
                }
                else if( field->at(x+y*resX) < thres && field->at(x+y*resX) >= 0.0f )
                {
                    dst.setPixel( x, y, sf::Color(0, 0, 0) );
                    break;
                }
                else
                {
                    float val = field->at(x+y*resX);
                    float indicator = std::fmod(val, static_cast<float>(1.0/col_st));

                    if( indicator > 0.0f && indicator < thresISO )
                    {
                        dst.setPixel( x, y, sf::Color(200, 200, 200) );
                        break;
                    }

                    if( i <= uField[x+y*resX] && uField[x+y*resX] <= i+col_st )
                    {
                        dst.setPixel( x, y, sf::Color(i+col_st, i+col_st, 0) );
                        break;
                    }
                }
            }
        }
    }

    std::string field_file_name = "isoline_field_" + fileName + ".jpg";
    dst.saveToFile( field_file_name );
    return dst;
}

} // namespace hfrep2D
