import AccuracyReport from './ui/AccuracyReport';

const VerificationSection = ({ verificationReport }) => {
  if (!verificationReport) return null;

  return (
    <AccuracyReport report={verificationReport} />
  );
};

export default VerificationSection;