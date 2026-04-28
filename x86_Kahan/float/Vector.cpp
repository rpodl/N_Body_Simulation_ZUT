#include "Vector.h"

std::istream& operator>>(std::istream& in, Vector& v) {
    return in >> v.px >> v.py >> v.pz;
}

std::ostream& operator<<(std::ostream& out, const Vector& v) {
    auto precision = out.precision();
    auto width = out.width();
    out << std::fixed << std::setw(width) << std::setprecision(precision) << v.px << "  ";
    out << std::fixed << std::setw(width) << std::setprecision(precision) << v.py << "  ";
    out << std::fixed << std::setw(width) << std::setprecision(precision) << v.pz;
    return out;
}
