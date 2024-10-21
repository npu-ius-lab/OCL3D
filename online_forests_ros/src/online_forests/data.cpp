#include <iostream>
#include <fstream>
#include <stdlib.h>

#include "online_forests/data.h"

using namespace std;

void DataSet::findFeatRange() {
        double minVal, maxVal;
        for (int i = 0; i < m_numFeatures; i++) {
                minVal = m_samples[0].x[i];
                maxVal = m_samples[0].x[i];
                for (int n = 1; n < m_numSamples; n++) {
                        if (m_samples[n].x[i] < minVal) {
                                minVal = m_samples[n].x[i];
                        }
                        if (m_samples[n].x[i] > maxVal) {
                                maxVal = m_samples[n].x[i];
                        }
                }

                m_minFeatRange.push_back(minVal);
                m_maxFeatRange.push_back(maxVal);
        }
}

void DataSet::loadLIBSVM(string filename) {
        ifstream fp(filename.c_str(), ios::binary);
        if (!fp) {
                cout << "Could not open input file " << filename << endl;
                exit(EXIT_FAILURE);
        }

        cout << "Loading data file: " << filename << " ... " << endl;

        // Reading the header
        int startIndex;
        fp >> m_numSamples;
        fp >> m_numFeatures;
        fp >> m_numClasses;
        fp >> startIndex;

        // Reading the data
        string line, tmpStr;
        int prePos, curPos, colIndex;
        m_samples.clear();

        for (int i = 0; i < m_numSamples; i++) {
                wsvector<double> x(m_numFeatures);
                Sample sample;
                resize(sample.x, m_numFeatures);
                fp >> sample.y; // read label
                sample.w = 1.0; // set weight

                getline(fp, line); // read the rest of the line
                prePos = 0;
                curPos = line.find(' ', 0);
                while (prePos <= curPos) {
                        prePos = curPos + 1;
                        curPos = line.find(':', prePos);
                        tmpStr = line.substr(prePos, curPos - prePos);
                        colIndex = atoi(tmpStr.c_str()) - startIndex;

                        prePos = curPos + 1;
                        curPos = line.find(' ', prePos);
                        tmpStr = line.substr(prePos, curPos - prePos);
                        x[colIndex] = atof(tmpStr.c_str());
                }
                copy(x, sample.x);
                m_samples.push_back(sample); // push sample into dataset
        }

        fp.close();

        if (m_numSamples != (int) m_samples.size()) {
                cout << "Could not load " << m_numSamples << " samples from " << filename;
                cout << ". There were only " << m_samples.size() << " samples!" << endl;
                exit(EXIT_FAILURE);
        }

        // Find the data range
        findFeatRange();

        cout << "Loaded " << m_numSamples << " samples with " << m_numFeatures;
        cout << " features and " << m_numClasses << " classes." << endl;
}

void DataSet::loadLIBSVM2(string data) {
        // Reading the header
        std::istringstream iss(data);
        string line;
        int startIndex;

        getline(iss, line, ' ');
        // std::cout << "96 " << line << endl;
        m_numSamples = atoi(line.c_str());
        getline(iss, line, ' ');
        m_numFeatures = atoi(line.c_str());
        getline(iss, line, ' ');
        m_numClasses = atoi(line.c_str());
        getline(iss, line, '\n');
        startIndex = atoi(line.c_str());


        // Reading the data
        string tmpStr;
        int prePos, curPos, colIndex;
        m_samples.clear();

        for (int i = 0; i < m_numSamples; i++) {
                wsvector<double> x(m_numFeatures);
                Sample sample;
                resize(sample.x, m_numFeatures);
                // getline(iss, line);
                // sample.y = atoi(line.substr(line.find(' ')).c_str()); // read label
                getline(iss, line, ' ');
                sample.y = atoi(line.c_str()); // read label
                // std::cout << "label is" <<sample.y << std::endl;
                sample.w = 1.0; // set weight

                getline(iss, line);
                prePos = 0;
                curPos = line.find(' ', 0);
                while (prePos <= curPos) {
                        prePos = curPos + 1;
                        curPos = line.find(':', prePos);
                        tmpStr = line.substr(prePos, curPos - prePos);
                        colIndex = atoi(tmpStr.c_str()) - startIndex;

                        prePos = curPos + 1;
                        curPos = line.find(' ', prePos);
                        tmpStr = line.substr(prePos, curPos - prePos);
                        x[colIndex] = atof(tmpStr.c_str());
                }
                copy(x, sample.x);
                m_samples.push_back(sample); // push sample into dataset
        }

        if (m_numSamples != (int) m_samples.size()) {
                cout << "Could not load " << m_numSamples;
                cout << ". There were only " << m_samples.size() << " samples!" << endl;
                exit(EXIT_FAILURE);
        }

        // Find the data range
        findFeatRange();

        cout << "Loaded " << m_numSamples << " samples with " << m_numFeatures;
        cout << " features and " << m_numClasses << " classes." << endl;
}


void DataSet::loadLIBSVM3(string data) {
        // Reading the header
        // std::cout << '154' << std::endl;
        std::istringstream iss(data);
        string line;
        int startIndex;

        getline(iss, line, ' ');
        // std::cout << "162 " << line << endl;
        m_numSamples = atoi(line.c_str());
        getline(iss, line, ' ');
        // std::cout << "165 " << line << endl;
        m_numFeatures = atoi(line.c_str());
        getline(iss, line, ' ');
        // std::cout << "168 " << line << endl;
        m_numClasses = atoi(line.c_str());
        getline(iss, line, '\n');
        // std::cout << "171 " << line << endl;
        startIndex = atoi(line.c_str());

        // Reading the data
        string tmpStr;
        int prePos, curPos, colIndex;
        m_samples.clear();
        // std::cout << '171' << std::endl;
        for (int i = 0; i < m_numSamples; i++) {
                wsvector<double> x(m_numFeatures);
                Sample sample;
                resize(sample.x, m_numFeatures);
                // getline(iss, line);
                // sample.y = atoi(line.substr(line.find(' ')).c_str()); // read label
                getline(iss, line, ' ');
                // std::cout << "186 " << line << endl;
                sample.y = atoi(line.c_str()); // read label
                // std::cout << "label is" <<sample.y << std::endl;
                sample.w = 1.0; // set weight

                getline(iss, line);
                // std::cout<< line << endl;
                //对于id,p_x,p_y,seq,stamp处理
                prePos = 0;
                curPos = line.find(' ', 0);
                tmpStr = line.substr(prePos, curPos - prePos);//id
                sample.id = atoi(tmpStr.c_str());
                // std::cout<< "id is" <<tmpStr <<  std::endl;
                //p_x
                prePos = curPos + 1;
                curPos = line.find(',', 0);
                tmpStr = line.substr(prePos, curPos - prePos);
                sample.position_x = atof(tmpStr.c_str());
                // std::cout<< "p_x is" <<tmpStr <<  std::endl;
                //p_y
                prePos = curPos + 1;
                curPos = line.find('*', 0);
                // std::cout<< "curPos is" <<curPos<< std::endl;
                tmpStr = line.substr(prePos, curPos - prePos);
                sample.position_y = atof(tmpStr.c_str());
                // std::cout<< "p_y is" <<tmpStr <<  std::endl;
                //seq
                prePos = curPos + 1;
                curPos = line.find('(', 0);
                // std::cout<< "curPos is" <<curPos<< std::endl;
                tmpStr = line.substr(prePos, curPos - prePos);
                sample.seq = atoi(tmpStr.c_str());
                // std::cout<< "seq is" <<tmpStr <<  std::endl;
                //stamp
                prePos = curPos + 1;
                curPos = line.find(')', 0);
                // std::cout<< "curPos is" <<curPos<< std::endl;
                tmpStr = line.substr(prePos, curPos - prePos);
                sample.stamp = atof(tmpStr.c_str());
                // std::cout<< "stamp is" <<tmpStr <<  std::endl;

                prePos = curPos;
                // curPos = line.find(' ', 0);
                while (prePos <= curPos) {
                        prePos = curPos + 1;
                        curPos = line.find(':', prePos);
                        tmpStr = line.substr(prePos, curPos - prePos);
                        colIndex = atoi(tmpStr.c_str()) - startIndex;

                        prePos = curPos + 1;
                        curPos = line.find(' ', prePos);
                        tmpStr = line.substr(prePos, curPos - prePos);
                        x[colIndex] = atof(tmpStr.c_str());
                }

                copy(x, sample.x);
                // std::cout<< "x is" <<sample.x <<  std::endl;
                m_samples.push_back(sample); // push sample into dataset
        }

        if (m_numSamples != (int) m_samples.size()) {
                cout << "Could not load " << m_numSamples;
                cout << ". There were only " << m_samples.size() << " samples!" << endl;
                exit(EXIT_FAILURE);
        }

        // Find the data range
        findFeatRange();

        cout << "Loaded " << m_numSamples << " samples with " << m_numFeatures;
        cout << " features and " << m_numClasses << " classes." << endl;
}
void DataSet::loadLIBSVM4(const pointnet_3d_box_stamped::PointNet3DBoxStampedArray& data) {
        // Reading the header
        // std::cout << '154' << std::endl;
        // std::istringstream iss(data);
        // string line;
        // int startIndex;

        // getline(iss, line, ' ');
        // // std::cout << "162 " << line << endl;

        m_numFeatures = data.fea_dimensions;
        m_numClasses = data.Classes;
        m_numSamples = data.number_of_samples;
        std::cout<<"m_numSamples " << m_numSamples<<std::endl;
   
        m_samples.clear();
        // std::cout << '171' << std::endl;
        for (int i = 0; i < m_numSamples; i++) {
                wsvector<double> x(m_numFeatures);
                Sample sample;
                resize(sample.x, m_numFeatures);
                std::string tmpStr = data.fea_boxes[i].label;
                sample.y = atoi(tmpStr.c_str());

                sample.header = data.header;
                
                sample.w = 1.0; // set weight


                sample.id = data.fea_boxes[i].id;

                sample.pose = data.fea_boxes[i].pose;
                // sample.position_x = data.fea_boxes[i].pose.position.x;
                // sample.position_y = data.fea_boxes[i].pose.position.y;
                // sample.position_z = data.fea_boxes[i].pose.position.z;    

                sample.dimensions = data.fea_boxes[i].dimensions;
                // sample.dimensions_x = data.fea_boxes[i].dimensions.x;
                // sample.dimensions_y = data.fea_boxes[i].dimensions.y;
                // sample.dimensions_z = data.fea_boxes[i].dimensions.z;  

               
                sample.header = data.fea_boxes[i].header;
                
                for(int j = 0;j < data.fea_boxes[i].features.size();j++){
                        x[j] = data.fea_boxes[i].features[j];
                        // std::cout << "x[j] is "<< " j " << j << " "<< x[j] << std::endl;
                }


                copy(x, sample.x);
                // sample.disp();
                m_samples.push_back(sample); // push sample into dataset
        }

        if (m_numSamples != (int) m_samples.size()) {
                cout << "Could not load " << m_numSamples;
                cout << ". There were only " << m_samples.size() << " samples!" << endl;
                exit(EXIT_FAILURE);
        }

        // Find the data range
        findFeatRange();

        cout << "Loaded " << m_numSamples << " samples with " << m_numFeatures;
        cout << " features and " << m_numClasses << " classes." << endl;
}