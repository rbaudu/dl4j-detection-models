package com.project.common.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * Classe utilitaire pour charger la configuration de l'application.
 * Cherche d'abord dans ./config/application.properties, puis dans les ressources
 * en cas d'échec.
 */
public class ConfigLoader {
    private static final Logger log = LoggerFactory.getLogger(ConfigLoader.class);
    private static final String CONFIG_FILE_PATH = "config/application.properties";
    private static final String DEFAULT_CONFIG_RESOURCE = "/default-config.properties";
    
    private ConfigLoader() {
        // Constructeur privé pour empêcher l'instanciation
    }
    
    /**
     * Charge la configuration depuis les fichiers de propriétés.
     * @return Properties contenant la configuration
     * @throws IOException si le chargement échoue
     */
    public static Properties loadConfiguration() throws IOException {
        Properties properties = new Properties();
        
        // Essayer de charger le fichier de configuration externe
        File configFile = new File(CONFIG_FILE_PATH);
        if (configFile.exists() && configFile.isFile()) {
            try (FileInputStream fis = new FileInputStream(configFile)) {
                properties.load(fis);
                log.info("Configuration chargée depuis {}", CONFIG_FILE_PATH);
                return properties;
            } catch (IOException e) {
                log.warn("Impossible de charger la configuration depuis {}, utilisation de la configuration par défaut", 
                         CONFIG_FILE_PATH, e);
            }
        } else {
            log.warn("Fichier de configuration {} non trouvé, utilisation de la configuration par défaut", 
                     CONFIG_FILE_PATH);
        }
        
        // Charger la configuration par défaut depuis les ressources
        try (InputStream is = ConfigLoader.class.getResourceAsStream(DEFAULT_CONFIG_RESOURCE)) {
            if (is != null) {
                properties.load(is);
                log.info("Configuration par défaut chargée depuis les ressources");
            } else {
                log.error("Impossible de trouver la configuration par défaut dans les ressources");
                throw new IOException("Configuration par défaut introuvable");
            }
        }
        
        return properties;
    }
    
    /**
     * Récupère une valeur de configuration comme String.
     * @param properties Propriétés chargées
     * @param key Clé de la propriété
     * @param defaultValue Valeur par défaut si la propriété n'est pas trouvée
     * @return Valeur de la propriété ou la valeur par défaut
     */
    public static String getString(Properties properties, String key, String defaultValue) {
        return properties.getProperty(key, defaultValue);
    }
    
    /**
     * Récupère une valeur de configuration comme int.
     * @param properties Propriétés chargées
     * @param key Clé de la propriété
     * @param defaultValue Valeur par défaut si la propriété n'est pas trouvée ou n'est pas un nombre
     * @return Valeur de la propriété comme int ou la valeur par défaut
     */
    public static int getInt(Properties properties, String key, int defaultValue) {
        String value = properties.getProperty(key);
        if (value == null) {
            return defaultValue;
        }
        
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            log.warn("La valeur de la propriété {} n'est pas un nombre valide, utilisation de la valeur par défaut", key);
            return defaultValue;
        }
    }
    
    /**
     * Récupère une valeur de configuration comme double.
     * @param properties Propriétés chargées
     * @param key Clé de la propriété
     * @param defaultValue Valeur par défaut si la propriété n'est pas trouvée ou n'est pas un nombre
     * @return Valeur de la propriété comme double ou la valeur par défaut
     */
    public static double getDouble(Properties properties, String key, double defaultValue) {
        String value = properties.getProperty(key);
        if (value == null) {
            return defaultValue;
        }
        
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            log.warn("La valeur de la propriété {} n'est pas un nombre valide, utilisation de la valeur par défaut", key);
            return defaultValue;
        }
    }
    
    /**
     * Récupère une valeur de configuration comme boolean.
     * @param properties Propriétés chargées
     * @param key Clé de la propriété
     * @param defaultValue Valeur par défaut si la propriété n'est pas trouvée
     * @return Valeur de la propriété comme boolean ou la valeur par défaut
     */
    public static boolean getBoolean(Properties properties, String key, boolean defaultValue) {
        String value = properties.getProperty(key);
        if (value == null) {
            return defaultValue;
        }
        
        return Boolean.parseBoolean(value);
    }
}
