/* A small structure to hold multiple return values for Iteration-AIS.
*/
public class PerIterationReturner {
        public PerIterationReturner(int[] z_, int[] topicCounts_, double result_) {
            z = z_;
            topicCounts = topicCounts_;
            result = result_;
        }
        public int[] z;
        public int[] topicCounts;
        public double result;
    }
