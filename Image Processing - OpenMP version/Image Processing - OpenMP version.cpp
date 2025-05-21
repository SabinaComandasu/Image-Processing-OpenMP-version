#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <vector>
#include <omp.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <string>
#include <cmath>
#include <chrono>
#include <direct.h>

void grayscale(unsigned char* image, int width, int height, int channels) {
    int img_size = width * height;
#pragma omp parallel for
    for (int i = 0; i < img_size; ++i) {
        int idx = i * channels;
        unsigned char r = image[idx];
        unsigned char g = image[idx + 1];
        unsigned char b = image[idx + 2];
        unsigned char gray = static_cast<unsigned char>(0.3 * r + 0.59 * g + 0.11 * b);
        image[idx] = image[idx + 1] = image[idx + 2] = gray;
    }
}

void invert(unsigned char* image, int width, int height, int channels) {
    int size = width * height * channels;
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        image[i] = 255 - image[i];
    }
}

void adjust_brightness(unsigned char* image, int width, int height, int channels, int value) {
    int size = width * height * channels;
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        int temp = image[i] + value;
        image[i] = static_cast<unsigned char>(std::max(0, std::min(255, temp)));
    }
}

void gaussian_blur(unsigned char* image, int width, int height, int channels) {
    std::vector<unsigned char> copy(image, image + width * height * channels);

    const float kernel[3][3] = {
        {1.f / 16, 2.f / 16, 1.f / 16},
        {2.f / 16, 4.f / 16, 2.f / 16},
        {1.f / 16, 2.f / 16, 1.f / 16}
    };

#pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int ix = (y + ky) * width + (x + kx);
                        sum += kernel[ky + 1][kx + 1] * copy[ix * channels + c];
                    }
                }
                image[(y * width + x) * channels + c] = static_cast<unsigned char>(sum);
            }
        }
    }
}

unsigned char* resize_custom(const unsigned char* image, int old_w, int old_h, int channels, int new_w, int new_h) {
    unsigned char* resized = new unsigned char[new_w * new_h * channels];

#pragma omp parallel for collapse(2)
    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            int src_x = x * old_w / new_w;
            int src_y = y * old_h / new_h;
            for (int c = 0; c < channels; ++c) {
                resized[(y * new_w + x) * channels + c] =
                    image[(src_y * old_w + src_x) * channels + c];
            }
        }
    }
    return resized;
}

template<typename Func>
void timed_operation(const std::string& label, Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "⏱ " << label << " completed in " << duration.count() << " ms\n";
}

int main() {
    std::string input_path;
    std::cout << "Enter image path (e.g., inputs/cat.jpeg): ";
    std::getline(std::cin, input_path);

    int width, height, channels;
    unsigned char* img = stbi_load(input_path.c_str(), &width, &height, &channels, 0);
    if (!img) {
        std::cerr << "❌ Failed to load image.\n";
        return 1;
    }

    std::cout << "✅ Loaded: " << width << "x" << height << " - " << channels << " channels\n";

    int choice;
    do {
        std::cout << "\n=== Image Processing Menu ===\n";
        std::cout << "1. Grayscale\n";
        std::cout << "2. Invert colors\n";
        std::cout << "3. Adjust brightness\n";
        std::cout << "4. Gaussian blur\n";
        std::cout << "5. Resize to custom size\n";
        std::cout << "6. Save and exit\n";
        std::cout << "Choose an option: ";
        std::cin >> choice;

        if (choice == 1) {
            timed_operation("Grayscale", [&]() {
                grayscale(img, width, height, channels);
                });
        }
        else if (choice == 2) {
            timed_operation("Invert", [&]() {
                invert(img, width, height, channels);
                });
        }
        else if (choice == 3) {
            int value;
            std::cout << "Enter brightness adjustment (-100 to 100): ";
            std::cin >> value;
            timed_operation("Brightness Adjustment", [&]() {
                adjust_brightness(img, width, height, channels, value);
                });
        }
        else if (choice == 4) {
            timed_operation("Gaussian Blur", [&]() {
                gaussian_blur(img, width, height, channels);
                });
        }
        else if (choice == 5) {
            int new_w, new_h;
            std::cout << "Enter new width in pixels: ";
            std::cin >> new_w;
            std::cout << "Enter new height in pixels: ";
            std::cin >> new_h;
            timed_operation("Resize", [&]() {
                unsigned char* resized = resize_custom(img, width, height, channels, new_w, new_h);
                stbi_image_free(img);
                img = resized;
                width = new_w;
                height = new_h;
                });
        }
        else if (choice == 6) {
            _mkdir("outputs");
            std::cin.ignore();  
            std::string filename;
            std::cout << "Enter output file name (e.g., result.jpg): ";
            std::getline(std::cin, filename);

            std::string output_path = "outputs/" + filename;
            bool success = stbi_write_jpg(output_path.c_str(), width, height, channels, img, 100);
            if (success) {
                std::cout << "✅ Image saved to: " << output_path << "\n";
            }
            else {
                std::cerr << "❌ Failed to save image.\n";
            }
        }

    } while (choice != 6);

    stbi_image_free(img);
    return 0;
}
