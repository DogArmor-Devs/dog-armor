// Railway infrastructure for DogArmor. Preview with `railway config plan`,
// apply with `railway config apply`. See https://docs.railway.com/infrastructure-as-code
import { defineRailway, github, project, service } from "railway/iac";

// This repository manages only its own resources in the environment.
export const partial = "web";

export default defineRailway(() => {
  const web = service("web", {
    source: github("DogArmor-Devs/dog-armor", { branch: "main" }),
    build: { builder: "DOCKERFILE", dockerfilePath: "Dockerfile" },
    deploy: { restartPolicyType: "ON_FAILURE", restartPolicyMaxRetries: 5 },
    healthcheck: "/healthz",
    healthcheckTimeout: 120,
    replicas: { iad: 1 },
  });

  return project("dog-armor", {
    // PORT and other service variables are set in the Railway dashboard.
    variables: { managed: false },
    resources: [web],
  });
});
