#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// Forward declaration updated to accept out_time_ms
void run_cuda_smith_waterman(const std::string& q, const std::string& d, 
                             int& out_score, int& out_start, int& out_stop, 
                             std::string& out_aligned_q, std::string& out_aligned_d,
                             float& out_time_ms);

struct SequencePair {
    std::string q;
    std::string d;
};

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (std::string::npos == first) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

std::vector<SequencePair> parseInputFile(const std::string& filename) {
    std::vector<SequencePair> pairs;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "'" << std::endl;
        return pairs;
    }

    std::string line;
    SequencePair currentPair;
    bool hasQ = false;

    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty()) continue;

        if (line.rfind("Q:", 0) == 0) { 
            currentPair.q = trim(line.substr(2));
            hasQ = true;
        } else if (line.rfind("D:", 0) == 0) { 
            currentPair.d = trim(line.substr(2));
            if (hasQ) {
                pairs.push_back(currentPair);
                hasQ = false;
            }
        }
    }
    file.close();
    return pairs;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file.txt>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    std::vector<SequencePair> dataset = parseInputFile(filename);

    if (dataset.empty()) {
        std::cerr << "No valid Q/D sequence pairs found in the file." << std::endl;
        return 1;
    }

    // Set up output file
    std::string out_filename = "outputs/sw_cuda.txt";
    std::ofstream outfile(out_filename);
    
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not create output file '" << out_filename << "'" << std::endl;
        return 1;
    }

    std::cout << "--- GPU Smith-Waterman Initialized ---" << std::endl;
    std::cout << "Processing " << dataset.size() << " pairs from " << filename << "..." << std::endl;

    // Process pairs and write to file
    for (size_t i = 0; i < dataset.size(); ++i) {
        int score = 0, start = 0, stop = 0;
        float gpu_time_ms = 0.0f;
        std::string aligned_q, aligned_d;
        
        // Execute GPU calculation
        run_cuda_smith_waterman_batched(dataset[i].q, dataset[i].d, score, start, stop, aligned_q, aligned_d, gpu_time_ms);
        
        // Write to file instead of console
        outfile << "Q:\t" << dataset[i].q << "\n";
        outfile << "D:\t" << dataset[i].d << "\n";
        outfile << "Match " << i + 1 << " [Score: " << score << ", Start: " << start << ", Stop: " << stop << "]\n";
        outfile << "\tD: " << aligned_d << "\n";
        outfile << "\tQ: " << aligned_q << "\n";
        outfile << "\t[GPU Matrix Compute Time]: " << gpu_time_ms << " ms\n";
        outfile << "----------------------------------------\n";
        
        // Optional: Print a progress dot to console for large files
        if ((i + 1) % 100 == 0) {
            std::cout << "Processed " << i + 1 << "/" << dataset.size() << " sequences..." << std::endl;
        }
    }

    outfile.close();
    std::cout << "Execution Complete! All results successfully saved to: " << out_filename << std::endl;

    return 0;
}