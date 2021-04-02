#ifndef TRACER_FIGURE_H
#define TRACER_FIGURE_H

struct Light {
    Light(const Vec3f &p, const Vec3f &c, const float &i) : position(p), color(c), intensity(i) {}
    Vec3f position;
    Vec3f color;
    float intensity;
};

struct Material {
    Material(const float &r, const Vec4f &a, const Vec3f &color, const float &spec) : refractive_index(r), albedo(a), diffuse_color(color), specular_exponent(spec) {}
    Material() : refractive_index(1), albedo(1,0,0,0), diffuse_color(), specular_exponent() {}
    float refractive_index;
    Vec4f albedo;
    Vec3f diffuse_color;
    float specular_exponent;
};

class Figure
{
public:
    explicit Figure(const Material &m) : material(m) {};
    virtual bool ray_intersect(const Vec3f &orig, const Vec3f &dir, float &t0) const {return false;};
    virtual Vec3f norm(const Vec3f &hit) const {return Vec3f(0, 0, 0);};

    Material material;
};

class Sphere : public Figure
{
public:
    Sphere(const Vec3f &c, const float &r, const Material &m) : Figure(m), center(c), radius(r) {}

    bool ray_intersect(const Vec3f &orig, const Vec3f &dir, float &t0) const
    {
        Vec3f L = center - orig;
        float tca = L*dir;
        float d2 = L*L - tca*tca;
        if (d2 > radius*radius) return false;
        float thc = sqrtf(radius*radius - d2);
        t0       = tca - thc;
        float t1 = tca + thc;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return false;
        return true;
    }

    Vec3f norm(const Vec3f &hit) const
    {
        return (hit - center).normalize();
    }

    Vec3f center;
    float radius;
};

class Triangle : public Figure
{
public:
    Triangle(const Vec3f &v0, const Vec3f &v1, const Vec3f &v2, const Material &m) : Figure(m),
            vertex0(v0), vertex1(v1), vertex2(v2)
    {
        normal = cross(vertex1 - vertex0, vertex2 - vertex0).normalize();
    }

    bool ray_intersect(const Vec3f &orig, const Vec3f &dir, float &t0) const
    {
        const float EPSILON = 0.0000001;
        Vec3f edge1, edge2, h, s, q;
        float a,f,u,v;
        edge1 = vertex1 - vertex0;
        edge2 = vertex2 - vertex0;
        h = cross(dir, edge2);
        a = edge1 * h;
        if (a > -EPSILON && a < EPSILON)
            return false;    // This ray is parallel to this triangle.
        f = 1.0/a;
        s = orig - vertex0;
        u = f * (s * h);
        if (u < 0.0 || u > 1.0)
            return false;
        q = cross(s, edge1);
        v = f * (dir * q);
        if (v < 0.0 || u + v > 1.0)
            return false;
        // At this stage we can compute t to find out where the intersection point is on the line.
        float t = f * (edge2 * q);
        if (t > EPSILON) // ray intersection
        {
            t0 = t;
            return true;
        }
        else // This means that there is a line intersection but not a ray intersection.
            return false;
    }

    Vec3f norm(const Vec3f &hit) const
    {
        return normal;
    }

    Vec3f vertex0;
    Vec3f vertex1;
    Vec3f vertex2;
    Vec3f normal;
};

#endif //TRACER_FIGURE_H
