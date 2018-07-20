import java.lang.Math;
import org.apache.commons.math3.special.Gamma;
import java.util.Random;

import cc.mallet.topics.MarginalProbEstimator;
import cc.mallet.types.*;

/* Code for the topic model evaluation algorithms in 
   James Foulds and Padhraic Smyth (2014).  Annealing Paths for the Evaluation of Topic Models. Proceedings of the Thirtieth Conference on Uncertainty in Artificial Intelligence (UAI 2014)
   @author James Foulds
*/
public class RatioAIS {

	/*Compute log-likelihood(document|topics1, tpAlpha1) - log-likelihood(document|topics2, tpAlpha2)
	using Ratio-AIS, with geometric averages of the two distributions as intermediate distributions.*/
	public static double RatioAIS_GeometricPath(int[] wordVector, double[][] topics1, double[] tpAlpha1, double[][] topics2, double[] tpAlpha2, int numImportanceSamples, int numTemperatures, int numBurnIn) {
				
		double[] logLikelihoodDifferencePerSample = new double[numImportanceSamples];
		double stepSize = 1.0/numTemperatures;
		int numTopics = tpAlpha1.length;

		for (int n = 0;n < numImportanceSamples; n++) {
			double logImportanceWeight = 0;
			
			//first sample from prior
			double[] theta_0;
			theta_0 = sampleFromDirichlet(tpAlpha2);
			int[] z;
			z = new int[wordVector.length];
			int[] topicCounts;
			topicCounts = new int[numTopics];
			for (int j = 0; j < wordVector.length; j++) {
				z[j] = sampleFromDiscrete(theta_0);
				topicCounts[z[j]]++;
			}
			
			double[] currentTopic1;
			double[] currentTopic2;
			
			//burn in to get a sample from parameter set 2.
			double[] pr_z_perTopic = new double[numTopics];
			for (int j = 0; j <numBurnIn; j++) {
				for (int i = 0; i < wordVector.length; i++) {
					//do gibbs update
					topicCounts[z[i]]--;
					double normConst = 0;
					currentTopic2 = topics2[wordVector[i]];
					for (int k = 0; k < numTopics; k++) {
						pr_z_perTopic[k] = currentTopic2[k] * (topicCounts[k] + tpAlpha2[k]);
						normConst += pr_z_perTopic[k];
					}
					for (int k = 0; k < numTopics; k++) {
						pr_z_perTopic[k] /= normConst; //normalize
					}
					z[i] = sampleFromDiscrete(pr_z_perTopic);
					topicCounts[z[i]]++;
				}
			}
			//record importance weight contribution of first sample
			for (int j = 0; j<wordVector.length; j++) {
				logImportanceWeight += Math.log(topics1[wordVector[j]][z[j]]) - Math.log(topics2[wordVector[j]][z[j]]);
			}
			//could be optimized: this is unnecessary if tpAlpha1 == tpAlpha2
			logImportanceWeight += polyaLogProb(topicCounts, tpAlpha1) - polyaLogProb(topicCounts, tpAlpha2);
			//System.out.println("finished burn in");
			//assert(stateIsConsistent(topicCounts, z));
			double temperature = 0;
			for (int s = 1; s <= numTemperatures-1; s++) {
				temperature += stepSize;
				for (int i = 0; i < wordVector.length; i++) {
					//do annealed gibbs update
					topicCounts[z[i]]--;
					double normConst = 0;
					currentTopic1 = topics1[wordVector[i]];
					currentTopic2 = topics2[wordVector[i]];
					for (int k = 0; k < numTopics; k++) {
						pr_z_perTopic[k] = Math.pow(currentTopic1[k] * (topicCounts[k] + tpAlpha1[k]), temperature)
										 * Math.pow(currentTopic2[k] * (topicCounts[k] + tpAlpha2[k]), 1-temperature);
						normConst += pr_z_perTopic[k];
					}
					for (int k = 0; k < numTopics; k++) {
						pr_z_perTopic[k] /= normConst; //normalize
					}
					z[i] = sampleFromDiscrete(pr_z_perTopic);
					topicCounts[z[i]]++;
				}
				for (int j = 0; j<wordVector.length; j++) {
					logImportanceWeight += Math.log(topics1[wordVector[j]][z[j]]) - Math.log(topics2[wordVector[j]][z[j]]);
				}
				logImportanceWeight += polyaLogProb(topicCounts, tpAlpha1) - polyaLogProb(topicCounts, tpAlpha2);
				
				//assert(stateIsConsistent(topicCounts, z));
			}
			logImportanceWeight *= stepSize; //when the steps sizes are uniformly spaced,
			//reweighting by p_s^(\tau_s - \tau_{s-1}) is the same as averaging over them (in log space).
			logLikelihoodDifferencePerSample[n] = logImportanceWeight;
			//System.out.println("" + logImportanceWeight);
		}
		double ll = sumLogProb(logLikelihoodDifferencePerSample) - Math.log(numImportanceSamples); //get the log of the sample average.
		return ll;
	}
	
	/*Compute log-likelihood(document|topics1, tpAlpha1) - log-likelihood(document|topics2, tpAlpha2)
	using Ratio-AIS, with convex combinations of the two sets of parameters as intermediate distributions.*/
	public static double RatioAIS_ConvexPath(int[] wordVector, double[][] topics1, double[] tpAlpha1, double[][] topics2, double[] tpAlpha2, int numImportanceSamples, int numTemperatures, int numBurnIn) {
			
		double[] logLikelihoodDifferencePerSample = new double[numImportanceSamples];
		double stepSize = 1.0/numTemperatures;
		int numTopics = tpAlpha1.length;
		int numWords = topics1.length;
		
		//Arrays to keep track of topics and alphas.  We need the second array to compute importance weights
		//Neal's notation is slightly weird, x_s (or z_s in our case) is sampled based on p_{s+1}.
		//This means that phi_s_plus_one is the phi we are currently drawing based on, and phi_s is the next phi we will draw from
		double[][] phi_s_plus_one = new double[numWords][numTopics];
		double[][] phi_s = new double[numWords][numTopics];
		double[] alpha_s_plus_one = new double[numTopics];
		double[] alpha_s = new double[numTopics];
		
		for (int n = 0;n < numImportanceSamples; n++) {
			//System.out.println("Importance sample " + n);
			double logImportanceWeight = 0;
			double temperature = 0; //these temperatures are "inverse temperatures" in the sense that we increase them instead of cool them.
			
			//assign phis and alphas
			for (int i = 0; i < numWords; i++) {
				for (int j = 0; j < numTopics; j++) {
					phi_s_plus_one[i][j] = topics2[i][j];
				}
			}
			assignPhi_s(phi_s, topics1, topics2, temperature + stepSize);
			for (int j = 0; j < numTopics; j++) {
				alpha_s_plus_one[j] = tpAlpha2[j];
			}
			assignAlpha_s(alpha_s, tpAlpha1, tpAlpha2, temperature + stepSize);
			

			
			//first sample from prior
			double[] theta_0;
			theta_0 = sampleFromDirichlet(tpAlpha2);
			int[] z;
			z = new int[wordVector.length];
			int[] topicCounts;
			topicCounts = new int[numTopics];
			for (int j = 0; j < wordVector.length; j++) {
				z[j] = sampleFromDiscrete(theta_0);
				topicCounts[z[j]]++;
			}
			
			double[] currentTopic;
			
			//burn in to get a sample from parameter set 2.
			double[] pr_z_perTopic = new double[numTopics];
			for (int j = 0; j <numBurnIn; j++) {
				for (int i = 0; i < wordVector.length; i++) {
					//do gibbs update
					topicCounts[z[i]]--;
					double normConst = 0;
					currentTopic = topics2[wordVector[i]];
					for (int k = 0; k < numTopics; k++) {
						pr_z_perTopic[k] = currentTopic[k] * (topicCounts[k] + tpAlpha2[k]);
						normConst += pr_z_perTopic[k];
					}
					for (int k = 0; k < numTopics; k++) {
						pr_z_perTopic[k] /= normConst; //normalize
					}
					z[i] = sampleFromDiscrete(pr_z_perTopic);
					topicCounts[z[i]]++;
				}
			}
			//record importance weight contribution of first sample
			for (int j = 0; j<wordVector.length; j++) {
				logImportanceWeight += Math.log(phi_s[wordVector[j]][z[j]]) - Math.log(phi_s_plus_one[wordVector[j]][z[j]]);
			}
			//could be optimized: this is unnecessary if tpAlpha1 == tpAlpha2
			logImportanceWeight += polyaLogProb(topicCounts, alpha_s) - polyaLogProb(topicCounts, alpha_s_plus_one);
			//System.out.println("finished burn in");
			//assert(stateIsConsistent(topicCounts, z));
			
			double[][] temp; //a temporary pointer for when we swap the phis
			double[] temp2; //a temporary pointer for when we swap the alphas
			
			for (int s = 1; s <= numTemperatures-1; s++) {
				temperature += stepSize;
				
				//Update topics and alphas to the current temperature
				temp = phi_s_plus_one; //keep its memory location so we don't have to reallocate			
				phi_s_plus_one = phi_s;
				phi_s = temp; //it now points to the memory of the previous phi_s_plus_one, but we will now overwrite it with the new values
				assignPhi_s(phi_s, topics1, topics2, temperature + stepSize);
				temp2 = alpha_s_plus_one;			
				alpha_s_plus_one = alpha_s;
				alpha_s = temp2;
				assignAlpha_s(alpha_s, tpAlpha1, tpAlpha2, temperature + stepSize);
				
				for (int i = 0; i < wordVector.length; i++) {
					//do gibbs update
					topicCounts[z[i]]--;
					double normConst = 0;
					currentTopic = phi_s_plus_one[wordVector[i]];
					for (int k = 0; k < numTopics; k++) {
						pr_z_perTopic[k] = currentTopic[k] * (topicCounts[k] + tpAlpha2[k]);
						normConst += pr_z_perTopic[k];
					}
					for (int k = 0; k < numTopics; k++) {
						pr_z_perTopic[k] /= normConst; //normalize
					}
					z[i] = sampleFromDiscrete(pr_z_perTopic);
					topicCounts[z[i]]++;
				}
				
				for (int j = 0; j<wordVector.length; j++) {
					logImportanceWeight += Math.log(phi_s[wordVector[j]][z[j]]) - Math.log(phi_s_plus_one[wordVector[j]][z[j]]);
				}
				//could be optimized: this is unnecessary if tpAlpha1 == tpAlpha2
				logImportanceWeight += polyaLogProb(topicCounts, alpha_s) - polyaLogProb(topicCounts, alpha_s_plus_one);
				
				//assert(stateIsConsistent(topicCounts, z));
			}

			logLikelihoodDifferencePerSample[n] = logImportanceWeight;
			//System.out.println("" + logImportanceWeight);
		}
		double ll = sumLogProb(logLikelihoodDifferencePerSample) - Math.log(numImportanceSamples); //get the log of the sample average.
		return ll;
	}
	
	/*A helper function to compute an intermediates phi for the convex path, and store it in phi_s.*/
	private static void assignPhi_s(double[][] phi_s, double[][] topics1, double[][] topics2, double temperature) {
		int numWords = topics1.length;
		int numTopics = topics1[1].length;
		
		double oneMinusTemp = 1 - temperature;
		for (int i = 0; i < numWords; i++) {
			for (int j = 0; j < numTopics; j++) {
				phi_s[i][j] = temperature * topics1[i][j] + oneMinusTemp * topics2[i][j];
			}
		}
	}
	
	/*A helper function to compute an intermediate alpha for the convex path, and store it in alpha_s.*/
	private static void assignAlpha_s(double[] alpha_s, double[] tpAlpha1, double[] tpAlpha2, double temperature) {
		int numTopics = tpAlpha1.length;
		double oneMinusTemp = 1 - temperature;
		
		for (int j = 0; j < numTopics; j++) {
			alpha_s[j] = temperature * tpAlpha1[j] + oneMinusTemp * tpAlpha2[j];
		}
	}

    /*Perform a segment of Iteration-AIS by annealing between two learned topic models.
      @returns a PerIterationReturner object, containing the final z vector, final topic counts
       and the difference in log-likelihood from the previous model in topics1.
    */
	public static PerIterationReturner IterationAIS_ConvexPath(int[] wordVector, double[][] topics1, double[] tpAlpha1, double[][] topics2, double[] tpAlpha2, int numTemperatures, int[] z, int[] topicCounts) {
		
		//the wordVector entries are one based, but in java they index the topics array which is now zero-based
		/*for (int i = 0; i < wordVector.length; i++) {
			wordVector[i]--;
		}*/
        int numImportanceSamples = 1;
        int numBurnIn = 0;		

		double[] logLikelihoodDifferencePerSample = new double[numImportanceSamples];
		double stepSize = 1.0/numTemperatures;
		int numTopics = tpAlpha1.length;
		int numWords = topics1.length;
		
		//Arrays to keep track of topics and alphas.  We need the second array to compute importance weights
		//Neal's notation is weird, x_s (or z_s in our case) is sampled based on p_{s+1}.
		//This means that phi_s_plus_one is the phi we are currently drawing based on, and phi_s is the next phi we will draw from
		double[][] phi_s_plus_one = new double[numWords][numTopics];
		double[][] phi_s = new double[numWords][numTopics];
		double[] alpha_s_plus_one = new double[numTopics];
		double[] alpha_s = new double[numTopics];
		
		for (int n = 0;n < numImportanceSamples; n++) {
			//System.out.println("Importance sample " + n);
			double logImportanceWeight = 0;
			double temperature = 0; //these temperatures are "inverse temperatures" in the sense that we increase them instead of cool them.
			
			//assign phis and alphas
			for (int i = 0; i < numWords; i++) {
				for (int j = 0; j < numTopics; j++) {
					phi_s_plus_one[i][j] = topics2[i][j];
				}
			}
			assignPhi_s(phi_s, topics1, topics2, temperature + stepSize);
			for (int j = 0; j < numTopics; j++) {
				alpha_s_plus_one[j] = tpAlpha2[j];
			}
			assignAlpha_s(alpha_s, tpAlpha1, tpAlpha2, temperature + stepSize);
			
			double[] currentTopic;
			
			//burn in to get a sample from parameter set 2.
			double[] pr_z_perTopic = new double[numTopics];
			for (int j = 0; j <numBurnIn; j++) {
				for (int i = 0; i < wordVector.length; i++) {
					//do gibbs update
					topicCounts[z[i]]--;
					double normConst = 0;
					currentTopic = topics2[wordVector[i]];
					for (int k = 0; k < numTopics; k++) {
						pr_z_perTopic[k] = currentTopic[k] * (topicCounts[k] + tpAlpha2[k]);
						normConst += pr_z_perTopic[k];
					}
					for (int k = 0; k < numTopics; k++) {
						pr_z_perTopic[k] /= normConst; //normalize
					}
					z[i] = sampleFromDiscrete(pr_z_perTopic);
					topicCounts[z[i]]++;
				}
			}
			//record importance weight contribution of first sample
			for (int j = 0; j<wordVector.length; j++) {
				logImportanceWeight += Math.log(phi_s[wordVector[j]][z[j]]) - Math.log(phi_s_plus_one[wordVector[j]][z[j]]);
			}
			//could be optimized: this is unnecessary if tpAlpha1 == tpAlpha2
			logImportanceWeight += polyaLogProb(topicCounts, alpha_s) - polyaLogProb(topicCounts, alpha_s_plus_one);
			//System.out.println("finished burn in");
			//assert(stateIsConsistent(topicCounts, z));
			
			double[][] temp; //a temporary pointer for when we swap the phis
			double[] temp2; //a temporary pointer for when we swap the alphas
			
			for (int s = 1; s <= numTemperatures-1; s++) {
				temperature += stepSize;
				
				//Update topics and alphas to the current temperature
				temp = phi_s_plus_one; //keep its memory location so we don't have to reallocate			
				phi_s_plus_one = phi_s;
				phi_s = temp; //it now points to the memory of the previous phi_s_plus_one, but we will now overwrite it with the new values
				assignPhi_s(phi_s, topics1, topics2, temperature + stepSize);
				temp2 = alpha_s_plus_one;			
				alpha_s_plus_one = alpha_s;
				alpha_s = temp2;
				assignAlpha_s(alpha_s, tpAlpha1, tpAlpha2, temperature + stepSize);
				
				for (int i = 0; i < wordVector.length; i++) {
					//do gibbs update
					topicCounts[z[i]]--;
					double normConst = 0;
					currentTopic = phi_s_plus_one[wordVector[i]];
					for (int k = 0; k < numTopics; k++) {
						pr_z_perTopic[k] = currentTopic[k] * (topicCounts[k] + tpAlpha2[k]);
						normConst += pr_z_perTopic[k];
					}
					for (int k = 0; k < numTopics; k++) {
						pr_z_perTopic[k] /= normConst; //normalize
					}
					z[i] = sampleFromDiscrete(pr_z_perTopic);
					topicCounts[z[i]]++;
				}
				
				for (int j = 0; j<wordVector.length; j++) {
					logImportanceWeight += Math.log(phi_s[wordVector[j]][z[j]]) - Math.log(phi_s_plus_one[wordVector[j]][z[j]]);
				}
				//could be optimized: this is unnecessary if tpAlpha1 == tpAlpha2
				logImportanceWeight += polyaLogProb(topicCounts, alpha_s) - polyaLogProb(topicCounts, alpha_s_plus_one);
				
				//assert(stateIsConsistent(topicCounts, z));
			}

			logLikelihoodDifferencePerSample[n] = logImportanceWeight;
			//System.out.println("" + logImportanceWeight);
		}
		double ll = sumLogProb(logLikelihoodDifferencePerSample) - Math.log(numImportanceSamples); //get the log of the sample average.
        PerIterationReturner ret = new PerIterationReturner(z, topicCounts, ll);
		return ret;
	}
    

    /*Compute log-likelihood(document|topics, tpAlpha) for the first iteration of Iteration-AIS, using Wallach et al.'s AIS algorithm.
      It returns the z's as well as the result.  Otherwise it's the same as LDALogLikelihood_AIS.*/
    public static PerIterationReturner IterationAIS_ConvexPath_FirstIteration(int[] wordVector, double[][] topics, double[] tpAlpha, int numTemperatures) {
		
		double stepSize = 1.0/numTemperatures;
		int numTopics = tpAlpha.length;

		double logImportanceWeight = 0;
		
		//first sample from prior
		double[] theta_0 = sampleFromDirichlet(tpAlpha);
		int[] z = new int[wordVector.length]; 
		int[] topicCounts = new int[numTopics];
		for (int j = 0; j < wordVector.length; j++) {
			z[j] = sampleFromDiscrete(theta_0);
			topicCounts[z[j]]++;
		}
		
		//record importance weight contribution of first sample
		for (int j = 0; j<wordVector.length; j++) {
			logImportanceWeight += Math.log(topics[wordVector[j]][z[j]]);
		}
		//System.out.println("initial sample weight " + logImportanceWeight);
		
		//assert(stateIsConsistent(topicCounts, z));
        double[] pr_z_perTopic = new double[numTopics];
		double temperature = 0;
		//for (int s = 0; s < numTemperatures-1; s++) {
		double[] currentTopic;
		for (int s = 1; s <= numTemperatures-1; s++) {
			temperature += stepSize;
			for (int i = 0; i < wordVector.length; i++) {
				//do annealed gibbs update
				topicCounts[z[i]]--;
				double normConst = 0;
				currentTopic = topics[wordVector[i]];
				for (int k = 0; k < numTopics; k++) {
					pr_z_perTopic[k] = Math.pow(currentTopic[k], temperature) * (topicCounts[k] + tpAlpha[k]);
					normConst += pr_z_perTopic[k];
				}
				for (int k = 0; k < numTopics; k++) {
					pr_z_perTopic[k] /= normConst; //normalize
				}
				z[i] = sampleFromDiscrete(pr_z_perTopic);
				topicCounts[z[i]]++;

			}
			for (int j = 0; j<wordVector.length; j++) {
				logImportanceWeight += Math.log(topics[wordVector[j]][z[j]]);
			}

			//assert(stateIsConsistent(topicCounts, z));
		}
		logImportanceWeight *= stepSize; //when the steps sizes are uniformly spaced,
		//reweighting by p_s^(\tau_s - \tau_{s-1}) is the same as averaging over them (in log space).
	
		double ll = logImportanceWeight;
        PerIterationReturner ret = new PerIterationReturner(z, topicCounts, ll);
		return ret;
	}


    /*Compute log-likelihood(document|topics, tpAlpha) using Radford Neal's annealed importance sampling,
      as described for topic models in Wallach et al. (2009), but with multiple importance samples.
      The procedure is to draw each initial sample from Pr(Z| tpAlpha),	then anneal to Pr(words, Z| topics, tpAlpha)
      \propto Pr(Z|words, topics, tpAlpha), and average the resulting importance weights.*/
    public static double LDALogLikelihood_AIS(int[] wordVector, double[][] topics, double[] tpAlpha, int numImportanceSamples, int numTemperatures) {
	
		//the wordVector entries are one based, but in java they index the topics array which is now zero-based
		/*for (int i = 0; i < wordVector.length; i++) {
			wordVector[i]--;
		}*/
		
		double[] logLikelihoodPerSample = new double[numImportanceSamples];
		double stepSize = 1.0/numTemperatures;
		int numTopics = tpAlpha.length;

		for (int n = 0;n < numImportanceSamples; n++) {
			double logImportanceWeight = 0;
			
			//first sample from prior
			double[] theta_0 = sampleFromDirichlet(tpAlpha);
			int[] z = new int[wordVector.length];
			int[] topicCounts = new int[numTopics];
			for (int j = 0; j < wordVector.length; j++) {
				z[j] = sampleFromDiscrete(theta_0);
				topicCounts[z[j]]++;
			}
			
			//record importance weight contribution of first sample
			for (int j = 0; j<wordVector.length; j++) {
				logImportanceWeight += Math.log(topics[wordVector[j]][z[j]]);
			}
			//System.out.println("initial sample weight " + logImportanceWeight);
			
			//assert(stateIsConsistent(topicCounts, z));
            double[] pr_z_perTopic = new double[numTopics];
			double temperature = 0;
			//for (int s = 0; s < numTemperatures-1; s++) {
			double[] currentTopic;
			for (int s = 1; s <= numTemperatures-1; s++) {
				temperature += stepSize;
				for (int i = 0; i < wordVector.length; i++) {
					//do annealed gibbs update
					topicCounts[z[i]]--;
					double normConst = 0;
					currentTopic = topics[wordVector[i]];
					for (int k = 0; k < numTopics; k++) {
						pr_z_perTopic[k] = Math.pow(currentTopic[k], temperature) * (topicCounts[k] + tpAlpha[k]);
						normConst += pr_z_perTopic[k];
					}
					for (int k = 0; k < numTopics; k++) {
						pr_z_perTopic[k] /= normConst; //normalize
					}
					z[i] = sampleFromDiscrete(pr_z_perTopic);
					topicCounts[z[i]]++;

				}
				for (int j = 0; j<wordVector.length; j++) {
					logImportanceWeight += Math.log(topics[wordVector[j]][z[j]]);
				}
			}
			logImportanceWeight *= stepSize; //when the steps sizes are uniformly spaced,
			//reweighting by p_s^(\tau_s - \tau_{s-1}) is the same as averaging over them (in log space).
			logLikelihoodPerSample[n] = logImportanceWeight;
		}
		double ll = sumLogProb(logLikelihoodPerSample) - Math.log(numImportanceSamples); //get the log of the sample average.
		return ll;
	}
	
		
	private static int sampleFromDiscrete(double[] probs) {
		//Draw a sample from a discrete distribution
		double temp = Math.random();
		double total = 0;
		for (int i = 0; i < probs.length; i++) {
			total += probs[i];
			if (temp < total) {
				return i;
			}
		}
		assert(false);
        return -1;
	}

	private static double[] sampleFromDirichlet(double[] dirParams) {
		//Draw a sample from a Dirichlet distribution
		double[] sample = new double[dirParams.length];

		double normConst = 0;
		for (int i = 0; i < sample.length; i++) {
			sample[i] = sampleGamma(dirParams[i], 1);
			normConst += sample[i];
		}
		for (int i = 0; i < sample.length; i++) {
			sample[i] /= normConst;
		}
		//TODO check for NaN
		return sample;
	}

	
	private static boolean stateIsConsistent(int[] topicCounts, int[] z) {
		//ensure cached counts of topic assignments are consistent with the actual topic assignments
		boolean result = true;    
		//check total counts of topics are equal to the number of words in all of the documents
		int sum = 0;
		for (int i = 0; i < topicCounts.length; i++) {
			sum += topicCounts[i];
		}
		if (sum != z.length) {
			System.out.println("inconsistent total in topicCounts\n");
			result = false;
		}
		
		int[] topicCounts2 = new int[topicCounts.length];
		for (int i = 0; i < z.length; i++) {
		    topicCounts2[z[i]]++;
		}
		for (int i = 0; i < topicCounts.length; i++) {
			if (topicCounts[i] !=  topicCounts2[i])
			result = false;
		}
		return result;
	}
	
	/*compute log probability (log-likelihood) of a vector of counts, given the
	  parameter vector alpha, for a Polya (compound Dirichlet) distribution.
      (ignores the combinatorial normalization constant at the front).*/
	private static double polyaLogProb(int[] counts, double[] alpha) {
		double alphaSum = 0;
		double countsSum = 0;
		for (int i = 0; i < alpha.length; i++) {
			alphaSum += alpha[i];
			countsSum += counts[i];
		}
		
		double lp = Gamma.logGamma(alphaSum) - Gamma.logGamma(countsSum + alphaSum);
		for (int i = 0; i < alpha.length; i++) {
			lp += Gamma.logGamma(counts[i] + alpha[i]) - Gamma.logGamma(alpha[i]);
		}
        return lp;
	}
	
	//source: http://vyshemirsky.blogspot.com/2007/11/sample-from-gamma-distribution-in-java.html
	private static Random rng = new Random(java.util.Calendar.getInstance().getTimeInMillis() + Thread.currentThread().getId());
	public static double sampleGamma(double k, double theta) {
		boolean accept = false;
		if (k < 1) {
			 // Weibull algorithm
			 double c = (1 / k);
			 double d = ((1 - k) * Math.pow(k, (k / (1 - k))));
			 double u, v, z, e, x;
			 do {
				  u = rng.nextDouble();
				  v = rng.nextDouble();
				  z = -Math.log(u);
				  e = -Math.log(v);
				  x = Math.pow(z, c);
				  if ((z + e) >= (d + x)) {
					accept = true;
				  }
			 } while (!accept);
			 return (x * theta);
		}
		else {
			 // Cheng's algorithm
			 double b = (k - Math.log(4));
			 double c = (k + Math.sqrt(2 * k - 1));
			 double lam = Math.sqrt(2 * k - 1);
			 double cheng = (1 + Math.log(4.5));
			 double u, v, x, y, z, r;
			 do {
				  u = rng.nextDouble();
				  v = rng.nextDouble();
				  y = ((1 / lam) * Math.log(v / (1 - v)));
				  x = (k * Math.exp(y));
				  z = (u * v * v);
				  r = (b + (c * y) - x);
				  if ((r >= ((4.5 * z) - cheng)) ||	(r >= Math.log(z))) {
					 accept = true;
				  }
			 } while (!accept);
			 return (x * theta);
		}
	}


   //THE FOLLOWING CODE IS BORROWED FROM MALLET!!! All license restrictions apply.
   private static final double LOGTOLERANCE = 30.0;
	/**
   * Sums an array of numbers log(x1)...log(xn).  This saves some of
   *  the unnecessary calls to Math.log in the two-argument version.
   * <p>
   * Note that this implementation IGNORES elements of the input
   *  array that are more than LOGTOLERANCE (currently 30.0) less
   *  than the maximum element.
   * <p>
   * Cursory testing makes me wonder if this is actually much faster than
   *  repeated use of the 2-argument version, however -cas.
   * @param vals An array log(x1), log(x2), ..., log(xn)
   * @return log(x1+x2+...+xn)
   */
  public static double sumLogProb (double[] vals)
  {
    double max = Double.NEGATIVE_INFINITY;
    int len = vals.length;
    int maxidx = 0;

    for (int i = 0; i < len; i++) {
      if (vals[i] > max) {
        max = vals[i];
        maxidx = i;
      }
    }

    boolean anyAdded = false;
    double intermediate = 0.0;
    double cutoff = max - LOGTOLERANCE;

    for (int i = 0; i < maxidx; i++) {
      if (vals[i] >= cutoff) {
        anyAdded = true;
        intermediate += Math.exp(vals[i] - max);
      }
    }
    for (int i = maxidx + 1; i < len; i++) {
      if (vals[i] >= cutoff) {
        anyAdded = true;
        intermediate += Math.exp(vals[i] - max);
      }
    }

    if (anyAdded) {
      return max + Math.log(1.0 + intermediate);
    } else {
      return max;
    }

  }


}
