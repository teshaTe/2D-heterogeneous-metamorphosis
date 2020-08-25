#ifndef DRAW_FIELD
#define DRAW_FIELD

#include <SFML/Graphics/RenderWindow.hpp>

namespace hfrep2D {

class render2D
{
public:
    render2D() { }
    ~render2D() { }

    void displayImage(sf::Image img);

    void convertFieldToImage(std::vector<unsigned char> *uField, const std::vector<float> *field,
                                      int resX, int resY );
    sf::Image drawIsolines(const std::vector<float> *field, int resX, int resY, float thres,
                       std::string fileName="", float thresISO = 0.0195f );

private:
    void saveImage(std::string fileName);
    void createWindow(int resX, int resY, std::string title);

    sf::RenderWindow *window;
};

} // namespace draw

#endif //define DRAW_FIELD
