package com.project.common.config;

import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Classe utilitaire pour résoudre les références de propriétés.
 * Permet de résoudre les variables de type ${property.name} dans les valeurs des propriétés.
 */
public class PropertyResolver {
    
    // Pattern pour trouver les références de propriétés ${property.name}
    private static final Pattern PROPERTY_PATTERN = Pattern.compile("\\$\\{([^}]*)\\}");
    
    /**
     * Résout toutes les références de propriétés dans une chaîne.
     * 
     * @param value La chaîne contenant des références de propriétés
     * @param properties Les propriétés à utiliser pour la résolution
     * @return La chaîne avec toutes les références résolues
     */
    public static String resolveProperty(String value, Properties properties) {
        if (value == null) {
            return null;
        }
        
        Matcher matcher = PROPERTY_PATTERN.matcher(value);
        StringBuilder result = new StringBuilder();
        int lastIndex = 0;
        
        while (matcher.find()) {
            // Ajouter le texte avant la référence
            result.append(value.substring(lastIndex, matcher.start()));
            
            // Récupérer le nom de la propriété référencée
            String propertyName = matcher.group(1);
            
            // Obtenir la valeur de la propriété
            String propertyValue = properties.getProperty(propertyName);
            
            // Si la propriété existe, l'utiliser, sinon garder la référence telle quelle
            if (propertyValue != null) {
                // Résoudre récursivement au cas où la propriété contient elle-même des références
                result.append(resolveProperty(propertyValue, properties));
            } else {
                // Garder la référence telle quelle
                result.append(matcher.group());
            }
            
            lastIndex = matcher.end();
        }
        
        // Ajouter le reste de la chaîne
        result.append(value.substring(lastIndex));
        
        return result.toString();
    }
    
    /**
     * Résout toutes les références de propriétés dans un ensemble de propriétés.
     * 
     * @param properties Les propriétés à résoudre
     * @return Les propriétés avec toutes les références résolues
     */
    public static Properties resolveProperties(Properties properties) {
        Properties resolved = new Properties();
        
        // Copier d'abord toutes les propriétés
        for (String name : properties.stringPropertyNames()) {
            resolved.setProperty(name, properties.getProperty(name));
        }
        
        // Résoudre ensuite toutes les références
        for (String name : resolved.stringPropertyNames()) {
            String value = resolved.getProperty(name);
            String resolvedValue = resolveProperty(value, resolved);
            resolved.setProperty(name, resolvedValue);
        }
        
        return resolved;
    }
}
