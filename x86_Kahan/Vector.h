#include <cmath>
#include <iomanip>
#include <iostream>

double sumGM(double sold, double a);

class Vector {
public:
    double px, py, pz;

    Vector() : px(0.0), py(0.0), pz(0.0) {}
    Vector(double x, double y, double z) : px(x), py(y), pz(z) {}

    double mod() const {
        return std::sqrt(px * px + py * py + pz * pz);
    }
    Vector operator+(const Vector& rhs) const {
        return Vector(px + rhs.px, py + rhs.py, pz + rhs.pz);
    }
    /*Vector operator+(const Vector& rhs) const {
        return Vector(sumGM(px, rhs.px), sumGM(py, rhs.py), sumGM(pz, rhs.pz));
    }*/
    Vector operator-(const Vector& rhs) const {
        return Vector(px - rhs.px, py - rhs.py, pz - rhs.pz);
    }
    Vector operator*(double s) const {
        return Vector(px * s, py * s, pz * s);
    }

    bool operator==(const Vector& rhs) const {
        return px == rhs.px && py == rhs.py && pz == rhs.pz;
    }

    friend std::istream& operator>>(std::istream&, Vector&);
    friend std::ostream& operator<<(std::ostream&, const Vector&);
};


