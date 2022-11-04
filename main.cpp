//
//  main.cpp
//  testopencv
//
//  Created by Ricardo Neftaly García King on 26/10/22.
//
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <vector>

using namespace std;
using namespace cv;


// Section of grayscale convertions
// Obtiene la imagen original y la imagen a escala de grises vacia, establece la imagen a escala de grises usando el promedio
void convertWithAverage(Mat& image, Mat& grayscale) {
    // Filas
    for (int i = 0; i < image.rows; i++) {
        // Columnas
        for (int j = 0; j < image.cols; j++) {
            grayscale.at<uchar>(Point(i,j)) = (uchar)( (image.at<Vec3b>(Point(i,j)).val[0] + image.at<Vec3b>(Point(i,j)).val[1] + image.at<Vec3b>(Point(i,j)).val[2]) / 3);
        }
    }
}

// Obtiene la imagen original y luego con coeficientes NTSC se convierte
void convertWithNTSC(Mat& image, Mat& grayscale) {
    // Rows
    for (int i = 0; i < image.rows; i++) {
        // Cols
        for (int j = 0; j < image.cols; j++) {
            grayscale.at<uchar>(Point(i, j)) = (uchar)( (0.299 * image.at<Vec3b>(Point(i, j)).val[0]) + (0.587 * image.at<Vec3b>(Point(i, j)).val[1]) + (0.11 * image.at<Vec3b>(Point(i, j)).val[2]) );
        }
    }
}


// Guardado de imagen
void saveImage(string path, Mat& image, string name){
    // Nombre de la imagen
    cout << "Tamanio de " + name + "\n";
    
    // Verificador de guardado de imagen
    if(imwrite(path, image)){
        cout << "\t\tColumnas: [" + to_string(image.cols) + "]. Filas: [" + to_string(image.rows) + "]" << endl;
    } else {
        cout << "Error en la ruta o en el guardado" << endl;
    }
}

// Kernel Gaussiano
Mat KernelCreation(int n, double sigma){
    // Calculos constantes
    double r, s = 2.0 * sigma * sigma;
    Mat GKernel(n,n, CV_64FC1);                                         // Matriz del kernel tamanio nxn

    // Sumatoria para normalizacion
    double sum = 0.0;
    
    // Factor de offset
    int offset = int( n / 2 );
    
    // Generacion del kernel nxn
    for (int x = offset * -1; x <= offset; x++) {                       // Recorrido por X con offset
        for (int y = offset * -1; y <= offset; y++) {                   // Recorrido por Y con offset
            r = sqrt(x * x + y * y);                                    // Se obtiene r
            // Calculo de cada punto de la matriz
            GKernel.at<double>(x+offset, y+offset) = (exp(-(r * r) / s)) / (M_PI * s);
            sum += GKernel.at<double>(x+offset, y + offset);
        }
    }

    // Normalizacion del kernel
    for (int i = 0; i < n; ++i)                                         // Recorrido por X
        for (int j = 0; j < n; ++j)                                     // Recordido por Y
            GKernel.at<double>(i, j) /= sum;                            // Se divide entre la sumatoria
            //GKernel[i][j] /= sum;
    
    return GKernel;                                                     // Regresa la matriz
}

// Obtiene n y valida para que sea siempre impar
int getN(){
    int n = 0;                                                          // Tamanio del kernel
    cout << "Introduce solo numeros impares" << endl;
   
    // Ciclo que atrapa al usuario
    while (true) {                                                      // Validador
        cout << "Tamano del kernel: ";
        cin >> n;
        
        if(n % 2 == 0){                                                 // Si es par
            cout << "No puedes escoger un numero par, intenta con un impar" << endl;
        } else {
            return n;
        }
    }
    return 0;
}
// Agrega bordes con (n - 1) / 2 && (m - 2) / 2
Mat padding(Mat img, int kwidth, int kheight){
    Mat tmp;                                                            // Matriz temporal
    img.convertTo(tmp, CV_64FC1);                                       // Conversion de imagen original a 64 bits, Float, 1 canal
    int pad_rows, pad_cols;                                             // Bordes extra
    // Calculo de bordes extras
    pad_rows = (kheight - 1) / 2;                                       // Operacion para renglones extra
    pad_cols = (kwidth - 1) / 2;                                        // Operacion para columnas extra
    
    // Creacion de la imagen con el padding agregado en 64 bits, Float, 1 canal
    Mat padded_image(Size(tmp.cols + 2 * pad_cols, tmp.rows + 2 * pad_rows), CV_64FC1, Scalar(0));
    // Se copia de la imagen original al que tiene bordes nuevos
    img.copyTo(padded_image(Rect(pad_cols, pad_rows, img.cols, img.rows)));
    
    return padded_image;                                                // Regresa la imagen con bordes
}


// Filtros
// Filro Gaussiano
// Obtiene la imagen con efecto gausiano, se le pasa el tamanio n, el valor sigma, imagen original y la imagen gaussiana a modificar
void OwnGaussian(Mat& image, Mat& image_blurred, int n, double sigma){
    // Variables
    Mat kernel;                                                         // Kernel del gaussiano
    kernel = KernelCreation(n, sigma);                                  // Obtencion del kernel, se le pasa sigma y n
    
    // Agrega padding a la imagen
    image_blurred = padding(image_blurred, n, n);
    
    // Imagen temporal inicializada con ceros
    Mat tempOutput = Mat::zeros(image.size(), CV_64FC1);                // 64 bits, Float, 1 canal
    
    // Rellenado de datos en la imagen
    for(int i = 0; i < image.rows; i++){                                // Reorrido en las filas
        for(int j = 0; j < image.cols; j++){                            // Recorrido en las columnas
            // Se guarda en cada pixel de la imagen temporal la sumatoria de los productos de los vecinos de nxn de la imagen original
            tempOutput.at<double>(i,j) = sum(kernel.mul(image_blurred(Rect(j, i, n, n)))).val[0];
        }
    }
    
    // Conversion del temporal a la imagen gaussiana con 8bits, uint, 1 canal
    tempOutput.convertTo(image_blurred, CV_8UC1);
}

// Filtro sobel
void OwnSobel(Mat& image, Mat& filtered_image, string output_path){
    // Matriz gradiente sobel
    Mat gx = (Mat_<double>(3,3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
    Mat gy = (Mat_<double>(3,3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    
    // Conversion a escala de grises
    convertWithNTSC(image, filtered_image);
    
    // Padding 3x3
    filtered_image = padding(filtered_image, 3, 3);
    
    // Ojo, creado de tamanio filtered_image y no con image
    Mat gx_image = Mat::zeros(filtered_image.size(), CV_64FC1);
    Mat gy_image = Mat::zeros(filtered_image.size(), CV_64FC1);
    Mat g_image = Mat::zeros(filtered_image.size(), CV_64FC1);
    
    // Obtencion con filtro para gx
    for(int i = 0; i < image.rows; i++){
        for (int j = 0; j < image.cols; j++){
            // Convolucion
            gx_image.at<double>(i,j) = sum(gx.mul(filtered_image(Rect(j, i, 3, 3)))).val[0];
        }
    }
    // Reconversion
    gx_image.convertTo(gx_image, CV_8UC1);
        
    // Guardado
    saveImage(output_path + "gx.jpg", gx_image, "gx");
    
    // Obtencion con filtro para gy
    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++){
            // Convolucion
            gy_image.at<double>(i,j) = sum(gy.mul(filtered_image(Rect(j,i, 3, 3)))).val[0];
        }
    }
    // Reconversion
    gy_image.convertTo(gy_image, CV_8UC1);
    
    // Guardado
    saveImage(output_path + "gy.jpg", gy_image, "gy");
    
    // Seccion de G
    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++){
            // Punto a punto con G = sqrt(pow(gx) + pow(gy))
            g_image.at<double>(i,j) = sqrt(pow(gx_image.at<double>(i,j),2) + pow(gy_image.at<double>(i,j), 2));
        }
    }
    
    // Reconversion
    g_image.convertTo(g_image, CV_8UC1);
    
    // Guardado
    saveImage(output_path + "g.jpg", g_image, "g");
    
    /*
    //Mat tempOutput = Mat::zeros(image.size(), CV_64FC1);
    // Relleno de datos
    for (int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++){
            // Convolucion
            tempOutput.at<double>(i,j) = sum(sobel_grad.mul(filtered_image(Rect(j, i, 3, 3)))).val[0];
        }
    }
     */
    
    /*
    // Reescalamiento
    Mat tempOutput2 = Mat::zeros(tempOutput.size(), CV_64FC1);
    double min = 0, max = 0;
    minMaxLoc(tempOutput, &min, &max);
    
    // Formula de la pendiente m = y2 - y1 / x2 - x1
    // Puntos
    double x1 = min, x2 = max, y1 = 0, y2 = max/2;
    double m = (y2 - y1) / (x2 - x1);
    //double m = ((max / 2) - 0) / (max - abs(min));
    
    // Despeje para obtener b en T(r) = mr + b, evaluamos con x1
    double b = x1 * m;
    // Pasamos por la matriz reescalando
    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.rows; j++){
            tempOutput2.at<double>(i,j) = (tempOutput.at<double>(i,j) * m) + b;
        }
    }
     */
    
    // Reconversion
    //tempOutput.convertTo(filtered_image, CV_8UC1);
    
    
}
/*
void equalizeHistogram(int* pdata, int width, int height, int max_val = 255)
{
    int total = width * height;
    int n_bins = max_val + 1;

    // Compute histogram
    vector<int> hist(n_bins, 0);
    for (int i = 0; i < total; ++i) {
        hist[pdata[i]]++;
    }

    // Build LUT from cumulative histrogram

    // Find first non-zero bin
    int i = 0;
    while (!hist[i]) ++i;

    if (hist[i] == total) {
        for (int j = 0; j < total; ++j) {
            pdata[j] = i;
        }
        return;
    }

    // Compute scale
    float scale = (n_bins - 1.f) / (total - hist[i]);

    // Initialize lut
    vector<int> lut(n_bins, 0);
    i++;

    int sum = 0;
    for (; i < hist.size(); ++i) {
        sum += hist[i];
        // the value is saturated in range [0, max_val]
        lut[i] = max(0, min(int(round(sum scale)), max_val));
    }

    // Apply equalization
    for (int i = 0; i < total; ++i) {
        pdata[i] = lut[pdata[i]];
    }
}
 */

int main(int argc, const char * argv[]) {
    // *** ADVERTENCIA, LEER PRIMERO *** ///
    // *** Profesor, no puedo usar el imshow por problemas de compatibilidad de opencv
    // *** así que guardaré las imagenes con la funcion saveImage que hice, por eso puse una variable
    // *** llamada output_path, solo se sustituye con la ruta dónde desea guardar, yo ya modifico
    // *** el nombre de la imagen de salida.
    
    // Tenia mis comentarios en inglés pero por miedo a que piense que lo saqué de internet
    // lo puse en inglés, mis funciones las dejé en inglés, pero le explico el funcionamiento línea
    // por línea para que no dude de mi.
    
    // *** Variables del main *** //
    int n = 0;                                                          // Tamanio del kernel
    // Ruta de entrada
    string input_path = "/Users/ricardoneftalygarciaking/Documents/GitHub/ESCOM-works/5th Semester/Artificial Vision/examen/original.jpg";
    // Ruta de salida
    string output_path = "/Users/ricardoneftalygarciaking/Documents/GitHub/ESCOM-works/5th Semester/Artificial Vision/examen/";
    double sigma = 0.0;                                                 // Valor para sigma
    
    // ********************* //
    
    
    
    // *** Imagenes *** ///
    // Apertura y guardado de imagen original
    Mat image = imread(input_path, cv::IMREAD_COLOR);
    // Escala de grises
    Mat NTSC_bw_image(image.rows, image.cols, CV_8UC1);                 // 8 bits, uint, 1 canal, NTSC
    // Imagen Gausiana
    Mat gaussian_image(image.rows, image.cols, CV_8UC1);                // 8 bits, uint, 1 canal, NTSC
    // Imagen ecualizada
    Mat ecualized_image(image.rows, image.cols, CV_8UC1);
    // Imagen G
    Mat g_image(image.rows, image.cols, CV_8UC1);
    
    // ********************* //
    
    // Validador de imagen
    if(image.empty()){
        cout << "Error al leer la imagen" << endl;
        return -1;                                                      // Finaliza el programa si no lee bien la imagen
    }
    
    // Pasos de evaluación
    // *** 1. Imagen original *** //
    saveImage(output_path + "imagen_original.jpg", image, "imagen original");
    // ********************* //
    
    
    // *** 2. Escala de grises *** //
    convertWithNTSC(image, NTSC_bw_image);                              // Obtencion de imagen en escala de grises por NTSC
    // Guardado de imagenes
    saveImage(output_path + "imagen_grises_NTSC.jpg", NTSC_bw_image, "imagen a escala de grises");
    // ******************** //
    
    
    // *** 3.Imagen Suavizada (Gaussiano) *** //
    n = getN();                                                         // Obtiene el tamanio del kernel para el gaussiano
    
    // Se usa escala de grises para el gaussiano
    convertWithNTSC(image, gaussian_image);
    
    // Obtencion de sigma
    cout << "Sigma: ";
    cin >> sigma;
    
    // Aplicacion de Filtro gaussiano
    OwnGaussian(image, gaussian_image, n, sigma);
    
    // Guardado de imagen
    saveImage(output_path + "blurred_image.jpg", gaussian_image, "imagen suavizada (Gaussiana)");
    // ******************* //
    
    // *** 4. Imagen Ecualizada *** //
    
    
    // ********************* //
    
    
    // *** 5. G, (sobel) *** //
    OwnSobel(image, g_image, output_path);
    // Guardado de imagen
    //saveImage(output_path + "imagen_sobel.jpg", g_image, "Imagen Sobel");
    
    
    return 0;
}
