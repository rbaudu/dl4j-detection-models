package com.project.training;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Écouteur d'itération d'entraînement qui journalise le score du modèle
 * à intervalles réguliers.
 */
public class ScoreIterationWithLoggingListener extends BaseTrainingListener {
    private static final Logger log = LoggerFactory.getLogger(ScoreIterationWithLoggingListener.class);
    
    private int printIterations = 10;
    private long startTime;
    private int iteration = 0;

    /**
     * Constructeur avec intervalle d'impression par défaut (10)
     */
    public ScoreIterationWithLoggingListener() {
        this(10);
    }

    /**
     * Constructeur avec intervalle d'impression personnalisé
     * 
     * @param printIterations Nombre d'itérations entre chaque impression du score
     */
    public ScoreIterationWithLoggingListener(int printIterations) {
        this.printIterations = printIterations;
        this.startTime = System.currentTimeMillis();
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if (printIterations <= 0) {
            printIterations = 1;
        }
        
        if (iteration % printIterations == 0) {
            this.iteration = iteration;
            double score = model.score();
            long currentTime = System.currentTimeMillis();
            long timeElapsed = currentTime - startTime;
            
            log.info("Époque {}, itération {}: score = {}, temps écoulé = {} ms", 
                    epoch, iteration, score, timeElapsed);
            
            // Réinitialiser le chronomètre pour l'intervalle suivant
            startTime = currentTime;
        }
    }
}
