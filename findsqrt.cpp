#include <iostream>
// #include <chrono>

int main(){
    // auto start = std::chrono::high_resolution_clock::now();
    auto num = 42.0f;
    auto precision = 1e-6f;
    auto diff = 1e6f;
    auto best_guess = 1.0f; 
    while(diff > precision){
        //diff = (best_guess * best_guess - 42)/(2*(best_guess));
        diff = 0.5 * (best_guess - 42/best_guess);
        best_guess -= diff;
        if(diff < 0){
            diff = -diff;
        }
    }
    // auto end = std::chrono::high_resolution_clock::now();
    std::cout << best_guess << "\n";
    // std::chrono::duration<double> runtime = end - start;
    // std::cout << "Took " << runtime.count() << " seconds" << "\n";
    return 0;
}