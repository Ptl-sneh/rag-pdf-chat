import logger from "./logger";

export const loggerInfo = (message: string): void => {
    logger.info('\n')
    logger.info(`=================================`);
    logger.info(`${message}`);
    logger.info(`=================================`);
};