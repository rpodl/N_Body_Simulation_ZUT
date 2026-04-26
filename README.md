Pliki Vector.h i Vector.cpp zawierają bardzo uproszczone kontery wykorzystywane dalej w kodzie.
Plik NBody.cpp zawiera symulacje problemu n-ciał metodą All-Pairs zaimplementowaną na procesorze z użyciem OpenMP.
Plik NBody.cu zawiera symulacje problemu n-ciał metodą All-Pairs zaimplementowaną na karcie graficznej z użyciem CUDA oraz symulacje problemu n-ciał metodą Barnes - Hut zaimplementowaną heterogeniczne na procesorze oraz karcie graficznej.

Sposób użycia 

W celu przetestowania programu na dużym zbiorze danych:
program uruchamiamy podając 3 parametry:
- liczbę ciał
- liczbę kroków
- Flagę C dla symulacji na procesorze, G dla symulacji na karcie graficznej oraz GB dla symulacji Barnes - Hut.

Pliki EnergyCheckCPU oraz EnergyCheckGPU służą do badania zużycia energetycznego poprzez narzędzia perf oraz nvidia-smi.

Sposób użycia

EnergyCheckCPU odapalamy podając dwa parametry:
- liczbę ciał.
- liczbę kroków.

EnergyCheckGPU odpalamy podając trzy parametry:
- liczbę ciał.
- liczbę kroków.
- Flagę G dla symulacji All - Pairs oraz GB dla symulacji Barnes - Hut.

W celu przetestowania programu pod względem dokładności obliczeniowej na przykładzie testu keplera:
program odpalamy z użyciem flagi -k
