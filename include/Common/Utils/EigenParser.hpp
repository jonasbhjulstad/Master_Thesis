#ifndef EIGEN_PARSER_HPP
#define EIGEN_PARSER_HPP

#include <Eigen/Dense>
#include <vector>
#include <fstream>

using namespace Eigen;

template <typename M>
M load_csv(const std::string &path)
{
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ','))
        {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    if (rows > 0)
    {
        return Map<M>(values.data());
    }
    // return Map<const M>(values.data(), rows, values.size()/rows);
    else
    {
        return M::Zero();
    }
}

#endif