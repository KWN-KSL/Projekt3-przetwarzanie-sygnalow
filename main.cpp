#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <matplot/matplot.h>
#include <AudioFile.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <filesystem>
#include <complex>
#include <algorithm>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace matplot;

enum SignalType {
    SINUS,
    COSINUS,
    PROSTOKATNY,
    PILOKSZTALTNY,
    TROJKATNY
};

std::vector<double> generateSignal(SignalType type, double frequency, double amplitude, double duration, double sampleRate) {
    int numSamples = static_cast<int>(duration * sampleRate);
    std::vector<double> samples(numSamples);
    double t = 0.0;
    double dt = 1.0 / sampleRate;
    for (int i = 0; i < numSamples; ++i) {
        switch (type) {
        case SINUS:
            samples[i] = amplitude * sin(2 * M_PI * frequency * t);
            break;
        case COSINUS:
            samples[i] = amplitude * cos(2 * M_PI * frequency * t);
            break;
        case PROSTOKATNY:
            samples[i] = amplitude * (std::fmod(t, 1 / frequency) < (1 / (2 * frequency)) ? 1 : -1);
            break;
        case PILOKSZTALTNY:
            samples[i] = amplitude * ((2 * std::fmod(t * frequency, 1.0)) - 1);
            break;
        case TROJKATNY:
            samples[i] = amplitude * (2 * std::fabs(2 * std::fmod(t * frequency, 1.0) - 1) - 1);
            break;
        }
        t += dt;
    }
    return samples;
}

void displaySignal(const std::vector<double>& samples, double sampleRate, size_t factor) {
    size_t numSamples = samples.size();
    size_t numDisplayedSamples = numSamples / factor;
    std::vector<double> displayedTime(numDisplayedSamples);
    std::vector<double> displayedSamples(numDisplayedSamples);

    for (size_t i = 0; i < numDisplayedSamples; ++i) {
        displayedTime[i] = static_cast<double>(i * factor) / sampleRate;
        displayedSamples[i] = samples[i * factor];
    }

    auto figure = matplot::figure(true);
    auto ax = figure->add_subplot(1, 1, 1);
    ax->plot(displayedTime, displayedSamples);

    double maxSample = *std::max_element(displayedSamples.begin(), displayedSamples.end());
    double minSample = *std::min_element(displayedSamples.begin(), displayedSamples.end());
    double padding = (maxSample - minSample) * 0.15;
    ax->ylim({ minSample - padding, maxSample + padding });

    xlabel("Czas (s)");
    ylabel("Amplituda");
    title("Sygnal");
    show();
}

void displayDFT(const std::vector<std::complex<double>>& transformedSamples, double sampleRate, size_t factor) {
    size_t N = transformedSamples.size();
    size_t numDisplayedSamples = N / factor;
    std::vector<double> frequency(numDisplayedSamples);
    std::vector<double> magnitude(numDisplayedSamples);

    for (size_t i = 0; i < numDisplayedSamples; ++i) {
        frequency[i] = static_cast<double>(i * factor) * sampleRate / N;
        magnitude[i] = std::abs(transformedSamples[i * factor]);
    }

    auto figure = matplot::figure(true);
    auto ax = figure->add_subplot(1, 1, 1);
    ax->plot(frequency, magnitude);
    xlabel("Czestotliwosc (Hz)");
    ylabel("Amplituda");
    title("Transformata DFT");
    show();
}

std::vector<double> loadAudioFile(const std::string& filename, int& sampleRate) {
    std::vector<double> audioSamples;
    AudioFile<double> audioFile;
    audioFile.load(filename);

    sampleRate = audioFile.getSampleRate();
    int numSamples = audioFile.getNumSamplesPerChannel();
    audioSamples.resize(numSamples);

    for (int i = 0; i < numSamples; ++i) {
        audioSamples[i] = audioFile.samples[0][i];
    }

    return audioSamples;
}

std::vector<std::complex<double>> DFT(const std::vector<double>& samples) {
    size_t N = samples.size();
    std::vector<std::complex<double>> transform(N);
    for (size_t k = 0; k < N; ++k) {
        std::complex<double> sum(0.0, 0.0);
        for (size_t n = 0; n < N; ++n) {
            double angle = 2 * M_PI * k * n / N;
            sum += samples[n] * std::polar(1.0, -angle);
        }
        transform[k] = sum;
    }
    return transform;
}

std::vector<double> IDFT(const std::vector<std::complex<double>>& samples) {
    size_t N = samples.size();
    std::vector<double> inverse(N);
    for (size_t n = 0; n < N; ++n) {
        std::complex<double> sum(0.0, 0.0);
        for (size_t k = 0; k < N; ++k) {
            double angle = 2 * M_PI * k * n / N;
            sum += samples[k] * std::polar(1.0, angle);
        }
        inverse[n] = sum.real() / N;
    }
    return inverse;
}

std::vector<double> thresholdSignal(const std::vector<double>& samples, double threshold) {
    std::vector<double> binarySamples(samples.size());
    std::transform(samples.begin(), samples.end(), binarySamples.begin(),
        [threshold](double sample) { return sample > threshold ? 1.0 : 0.0; });
    return binarySamples;
}

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    py::enum_<SignalType>(m, "SignalType")
        .value("SINUS", SignalType::SINUS)
        .value("COSINUS", SignalType::COSINUS)
        .value("PROSTOKATNY", SignalType::PROSTOKATNY)
        .value("PILOKSZTALTNY", SignalType::PILOKSZTALTNY)
        .value("TROJKATNY", SignalType::TROJKATNY)
        .export_values();
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: scikit_build_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";
    m.def("generateSignal", &generateSignal, py::arg("type"), py::arg("frequency"), py::arg("amplitude"), py::arg("duration"), py::arg("sampleRate"));
    m.def("displaySignal", &displaySignal, py::arg("samples"), py::arg("sampleRate"), py::arg("factor"));
    m.def("displayDFT", &displayDFT, py::arg("transformedSamples"), py::arg("sampleRate"), py::arg("factor"));
    m.def("loadAudioFile", &loadAudioFile, py::arg("filename"), py::arg("sampleRate"));
    m.def("DFT", &DFT, py::arg("samples"));
    m.def("IDFT", &IDFT, py::arg("samples"));
    m.def("thresholdSignal", &thresholdSignal, py::arg("samples"), py::arg("threshold"));
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}