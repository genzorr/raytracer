#include <iostream>
#include <fstream>
#include "mpi.h"
#include "geometry.h"
#include "Figure.h"
//#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int WIDTH = 1024;
const int HEIGHT = 768;
const int DEPTH = 3;
char *FILENAME = "ivory_glass_1.jpg";

/// References:
/// https://habr.com/ru/post/436790/
/// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection

void SaveImage(char* filename, int w, int h, unsigned char* data)
{
    const int comp = 3;
    stbi_write_jpg(filename, w, h, comp, data, 100);
}

Vec3f reflect(const Vec3f &I, const Vec3f &N) {
    return I - N*2.f*(I*N);
}

Vec3f refract(const Vec3f &I, const Vec3f &N, const float &refractive_index) { // Snell's law
    float cosi = - std::max(-1.f, std::min(1.f, I*N));
    float etai = 1, etat = refractive_index;
    Vec3f n = N;
    if (cosi < 0) { // if the ray is inside the object, swap the indices and invert the normal to get the correct result
        cosi = -cosi;
        std::swap(etai, etat); n = -N;
    }
    float eta = etai / etat;
    float k = 1 - eta*eta*(1 - cosi*cosi);
    return k < 0 ? Vec3f(0,0,0) : I*eta + n*(eta * cosi - sqrtf(k));
}


bool scene_intersect(const Vec3f &orig, const Vec3f &dir, const std::vector<Figure*> &figures, Vec3f &hit, Vec3f &N, Material &material) {
    float figures_dist = std::numeric_limits<float>::max();
    for (auto &figure : figures)
    {
        float dist_i;
        if (figure->ray_intersect(orig, dir, dist_i) && dist_i < figures_dist)
        {
            figures_dist = dist_i;
            hit = orig + dir*dist_i;
            N = figure->norm(hit);
            material = figure->material;
        }
    }
    return figures_dist<1000;
}

Vec3f cast_ray(const Vec3f &orig, const Vec3f &dir, const std::vector<Figure*> &figures, const std::vector<Light> &lights, size_t depth=0) {
    Vec3f point, N;
    Material material;

    if (depth>DEPTH || !scene_intersect(orig, dir, figures, point, N, material)) {
        return Vec3f(0.2, 0.2, 0.2); // background color
//        return Vec3f(0.2, 0.7, 0.8); // background color
    }

    Vec3f reflect_dir = reflect(dir, N).normalize();
    Vec3f refract_dir = refract(dir, N, material.refractive_index).normalize();
    Vec3f reflect_orig = reflect_dir*N < 0 ? point - N*1e-3 : point + N*1e-3; // offset the original point to avoid occlusion by the object itself
    Vec3f refract_orig = refract_dir*N < 0 ? point - N*1e-3 : point + N*1e-3;
    Vec3f reflect_color = cast_ray(reflect_orig, reflect_dir, figures, lights, depth + 1);
    Vec3f refract_color = cast_ray(refract_orig, refract_dir, figures, lights, depth + 1);

    Vec3f color(0,0,0);
    float diffuse_light_intensity = 0, specular_light_intensity = 0;
    for (auto light : lights)
    {
        Vec3f light_dir      = (light.position - point).normalize();
        float light_distance = (light.position - point).norm();

        Vec3f shadow_orig = light_dir*N < 0 ? point - N*1e-3 : point + N*1e-3; // checking if the point lies in the shadow of the lights[i]
        Vec3f shadow_pt, shadow_N;
        Material tmpmaterial;
        if (scene_intersect(shadow_orig, light_dir, figures, shadow_pt, shadow_N, tmpmaterial) && (shadow_pt - shadow_orig).norm() < light_distance)
            continue;

        diffuse_light_intensity  = light.intensity * std::max(0.f, light_dir*N);
        specular_light_intensity = powf(std::max(0.f, -reflect(-light_dir, N)*dir), material.specular_exponent)*light.intensity;

        Vec3f tcolor = material.diffuse_color * diffuse_light_intensity * material.albedo[0] + Vec3f(1., 1., 1.)*specular_light_intensity * material.albedo[1] + reflect_color*material.albedo[2] + refract_color*material.albedo[3];
        tcolor[0] *= light.color[0];
        tcolor[1] *= light.color[1];
        tcolor[2] *= light.color[2];
        color = color + tcolor;
    }
    return color;
}

void render(int argc, char* argv[], const std::vector<Figure*> &figures, const std::vector<Light> &lights)
{
    const int   width    = WIDTH;
    const int   height   = HEIGHT;
    const float fov      = M_PI/3.;
    Vec3f camera(-1, 1.5, 0);
    std::vector<Vec3f> framebuffer(width*height);

    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double time;

    if (rank == 0)
    {
        std::cout << "Size: " << size << std::endl;
        std::cout << "Triangles: " << figures.size() << std::endl;
        time = MPI_Wtime();
    }

    for (size_t j = height / size * rank; j < height / size * (rank + 1); j++)
    { // actual rendering loop
        for (size_t i = 0; i < width; i++)
        {
            float dir_x = (i + 0.5) - width / 2.;
            float dir_y = -(j + 0.5) + height / 2.;    // this flips the image at the same time
            float dir_z = -height / (2. * tan(fov / 2.));
            framebuffer[i + j * width] = cast_ray(camera, (Vec3f(dir_x, dir_y, dir_z) - camera).normalize(), figures, lights);
        }
    }

    auto *data = new unsigned char[3*width*height];
    for (int i = 0; i < width*height; i+=1)
    {
        Vec3f &c = framebuffer[i];
        float max = std::max(c[0], std::max(c[1], c[2]));
        if (max>1) c = c*(1./max);
        for (int k = 0; k < 3; k++)
            data[3*i+k] = (char)(255 * std::max(0.f, std::min(1.f, framebuffer[i][k])));
    }

    MPI_Gather(&data[height / size * rank * width * 3], height * width / size * 3, MPI_CHAR, data, height * width / size * 3, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "Time: " << MPI_Wtime() - time << std::endl;
        SaveImage(FILENAME, width, height, data);
    }
    delete [] data;
    MPI_Finalize();
}

int ReadTriangles(const std::string &filename, const Material &m, std::vector<Figure*> &triangles, int points)
{
    std::ifstream file(filename);
    if (!file.is_open())
        return -1;
    Vec3f a,b,c;

    while (file && points--)
    {
        file >> a.x >> a.y >> a.z;
        file >> b.x >> b.y >> b.z;
        file >> c.x >> c.y >> c.z;
        triangles.emplace_back(new Triangle(a, b, c, m));
    }
    return 0;
}

int main(int argc, char* argv[])
{
    Material      ivory(1.0, Vec4f(0.6,  0.3, 0.1, 0.0), Vec3f(0.4, 0.4, 0.3),   50.);
    Material      glass(1.0, Vec4f(0.0,  0.5, 0.1, 0.8), Vec3f(0.6, 0.7, 0.8),  125.);
    Material red_rubber(1.0, Vec4f(0.9,  0.1, 0.0, 0.0), Vec3f(0.3, 0.1, 0.1),   10.);
    Material     mirror(1.0, Vec4f(0.0, 10.0, 0.8, 0.0), Vec3f(1.0, 1.0, 1.0), 1425.);

    /// Spheres test.
//    std::vector<Figure*> figures;
//    figures.emplace_back(new Sphere(Vec3f(-3,    0,   -16), 2,      ivory));
//    figures.emplace_back(new Sphere(Vec3f(-1.0, -1.5, -12), 2, red_rubber));
//    figures.emplace_back(new Sphere(Vec3f( 1.5, -0.5, -18), 3, red_rubber));
//    figures.emplace_back(new Sphere(Vec3f( 7,    5,   -18), 4,      ivory));

    /// Triangles test.
    std::vector<Figure*> figures;
    if (ReadTriangles("../src/model_rotated.txt", ivory, figures, 8664))
        return -1;
    figures.emplace_back(new Sphere(Vec3f(1, 1.5, -7), 1, glass));
    std::cout << "Read file\n";

    std::vector<Light>  lights;
    lights.emplace_back(Light(Vec3f(0, 3,  5), Vec3f(1, 0, 0), 2));
    lights.emplace_back(Light(Vec3f( 30, 50, -25), Vec3f(0, 1, 0), 4));
//    lights.emplace_back(Light(Vec3f( 30, 20,  30), 1.7));

    render(argc, argv, figures, lights);

    return 0;
}