Pliki Vector.h i Vector.cpp zawierają bardzo uproszczone kontery wykorzystywane dalej w kodzie.
Plik NBody.cpp zawiera symulacje problemu n-ciał metodą All-Pairs zaimplementowaną na procesorze z użyciem OpenMP.
Plik NBody.cu zawiera symulacje problemu n-ciał metodą All-Pairs zaimplementowaną na karcie graficznej z użyciem CUDA oraz symulacje problemu n-ciał metodą Barnes - Hut zaimplementowaną heterogeniczne na procesorze oraz karcie graficznej.

Sposób użycia 
program uruchamiamy podając 3 parametry:
- liczbę ciał
- liczbę kroków
- Flagę C dla symulacji na procesorze, G dla symulacji na karcie graficznej oraz GB dla symulacji Barnes - Hut.

