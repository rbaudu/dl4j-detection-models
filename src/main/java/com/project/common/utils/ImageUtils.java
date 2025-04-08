package com.project.common.utils;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * Classe utilitaire pour le traitement des images.
 * Fournit des méthodes pour charger et prétraiter les images pour différents modèles.
 */
public class ImageUtils {
    private static final Logger log = LoggerFactory.getLogger(ImageUtils.class);
    
    // Dimensions standards pour MobileNetV2
    public static final int MOBILENET_WIDTH = 224;
    public static final int MOBILENET_HEIGHT = 224;
    public static final int MOBILENET_CHANNELS = 3;
    
    private ImageUtils() {
        // Constructeur privé pour empêcher l'instanciation
    }
    
    /**
     * Charge et prétraite une image pour le modèle MobileNetV2.
     * - Redimensionne l'image à 224x224 pixels
     * - Normalise les valeurs entre -1 et 1
     * - Réorganise les dimensions au format attendu par MobileNetV2
     *
     * @param imageFile Fichier image à charger
     * @return INDArray contenant l'image prétraitée
     * @throws IOException en cas d'erreur lors du chargement de l'image
     */
    public static INDArray preprocessImageForMobileNetV2(File imageFile) throws IOException {
        log.debug("Prétraitement de l'image {} pour MobileNetV2", imageFile.getName());
        
        // Charger l'image et la redimensionner à 224x224 pixels (taille requise par MobileNetV2)
        NativeImageLoader loader = new NativeImageLoader(MOBILENET_HEIGHT, MOBILENET_WIDTH, MOBILENET_CHANNELS);
        INDArray image = loader.asMatrix(imageFile);
        
        // Normaliser les valeurs des pixels entre -1 et 1 (prétraitement standard pour MobileNetV2)
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(-1, 1);
        scaler.transform(image);
        
        // Réorganiser les dimensions si nécessaire (NCHW -> NHWC)
        // image = image.permute(0, 2, 3, 1);
        
        return image;
    }
    
    /**
     * Charge et prétraite un lot d'images pour le modèle MobileNetV2.
     *
     * @param imageFiles Tableau de fichiers images à charger
     * @return INDArray contenant les images prétraitées (forme [batchSize, height, width, channels])
     * @throws IOException en cas d'erreur lors du chargement des images
     */
    public static INDArray preprocessImagesForMobileNetV2(File[] imageFiles) throws IOException {
        if (imageFiles == null || imageFiles.length == 0) {
            throw new IllegalArgumentException("Le tableau de fichiers images ne peut pas être vide");
        }
        
        log.debug("Prétraitement d'un lot de {} images pour MobileNetV2", imageFiles.length);
        
        NativeImageLoader loader = new NativeImageLoader(MOBILENET_HEIGHT, MOBILENET_WIDTH, MOBILENET_CHANNELS);
        INDArray batchArray = loader.asMatrix(imageFiles);
        
        // Normaliser les valeurs des pixels entre -1 et 1
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(-1, 1);
        scaler.transform(batchArray);
        
        return batchArray;
    }
    
    /**
     * Vérifie si un fichier est une image valide basée sur son extension.
     *
     * @param file Fichier à vérifier
     * @return true si le fichier est une image, false sinon
     */
    public static boolean isImageFile(File file) {
        if (file == null || !file.isFile()) {
            return false;
        }
        
        String name = file.getName().toLowerCase();
        return name.endsWith(".jpg") || name.endsWith(".jpeg") || 
               name.endsWith(".png") || name.endsWith(".bmp") || 
               name.endsWith(".gif");
    }
}
