from time import process_time
import logging

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from PreProcessor import PreProcessor

logging.basicConfig(
    format="[%(levelname)s] %(asctime)s - %(module)s: %(message)s",
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


x, y = PreProcessor.preprocess_reviews("properties_template.py")

logger.info("Training model.")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
gbr = GradientBoostingRegressor(random_state=1)
fit_start = process_time()
gbr.fit(X_train, y_train)
fit_end = process_time()
logger.info(f"Fit time: {fit_end - fit_start}s")

logger.info("Testing model")
pred_start = process_time()
y_pred = gbr.predict(X_test)
pred_end = process_time()
logger.info(f"Prediction time: {pred_end - pred_start}s")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

logger.info(f"MSE: {mse}")
logger.info(f"R-Squared: {r2}")
