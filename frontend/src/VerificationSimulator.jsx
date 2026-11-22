
import { useState } from "react";

import CaseDashboard from "./CaseDashboard";

import "./VerificationSimulator.css";

const API_URL = "http://127.0.0.1:8000";
const STORAGE_KEY = "vaanishield_demo_last_case_id";

const actionNames = {
  balance_inquiry: "Balance Enquiry",
  transaction_status: "Transaction Status",
  password_reset: "Password Reset",
  mobile_number_change: "Mobile Number Change",
  transfer_limit_increase: "Transfer Limit Increase",
  high_value_transfer: "High Value Transfer",
};

async function apiRequest(path, options = {}) {
  const response = await fetch(
    `${API_URL}${path}`,
    options
  );

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(
      typeof data?.detail === "string"
        ? data.detail
        : `Request failed (HTTP ${response.status})`
    );
  }

  return data;
}

function saveLastCaseId(caseId) {
  try {
    localStorage.setItem(STORAGE_KEY, caseId);
  } catch {
    // SQLite remains the source of truth.
  }
}

function getLastCaseId() {
  try {
    return localStorage.getItem(STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

export default function VerificationSimulator({
  analysis,
}) {
  const [caseRecord, setCaseRecord] = useState(null);

  const [lookupId, setLookupId] = useState(
    getLastCaseId
  );

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [refreshToken, setRefreshToken] = useState(0);

  const context = analysis?.banking_context;
  const risk = analysis?.risk_assessment;

  const needsVerification =
    risk &&
    risk.action !== "STANDARD_BANKING_FLOW";


  function refreshDashboard() {
    setRefreshToken((value) => value + 1);
  }

  async function createCase() {
    if (
      !context ||
      !needsVerification ||
      loading
    ) {
      return;
    }

    setLoading(true);
    setError("");
    setMessage("");

    try {
      const data = await apiRequest(
        "/api/verification-cases",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            banking_action: context.requested_action,
            authentication_status:
              context.authentication_status,
            unusual_activity:
              context.unusual_activity,
          }),
        }
      );

      setCaseRecord(data);
      setLookupId(data.case_id);

      saveLastCaseId(data.case_id);

      setMessage(
        "Verification case created in SQLite."
      );

      refreshDashboard();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadCase() {
    const caseId = lookupId.trim();

    if (!/^[a-f0-9]{32}$/i.test(caseId)) {
      setError(
        "Enter a valid 32-character case ID."
      );
      return;
    }

    setLoading(true);
    setError("");
    setMessage("");

    try {
      const data = await apiRequest(
        `/api/verification-cases/${caseId}`
      );

      setCaseRecord(data);
      setLookupId(data.case_id);

      saveLastCaseId(data.case_id);

      setMessage(
        "Case loaded successfully from SQLite."
      );
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function respond(response) {
    if (
      !caseRecord ||
      caseRecord.status !==
        "PENDING_CUSTOMER_RESPONSE" ||
      loading
    ) {
      return;
    }

    setLoading(true);
    setError("");
    setMessage("");

    try {
      const data = await apiRequest(
        `/api/verification-cases/${caseRecord.case_id}/respond`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ response }),
        }
      );

      setCaseRecord(data);

      setMessage(
        response === "confirm"
          ? "Customer confirmation saved in SQLite."
          : "Customer denial saved in SQLite."
      );

      refreshDashboard();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleDashboardCaseUpdated(updatedCase) {
    setCaseRecord((current) => {
      if (
        current?.case_id === updatedCase.case_id
      ) {
        return updatedCase;
      }

      return current;
    });
  }

  const status = caseRecord?.status;

  const outcomeTone =
    status === "DISPUTED" ||
    status === "ESCALATED_FOR_INVESTIGATION"
      ? "danger"
      : status ===
        "REVIEW_CLOSED_NO_BANKING_ACTION"
      ? "success"
      : "warning";

  function outcomeTitle() {
    switch (status) {
      case "DISPUTED":
        return "Request disputed";

      case "CONFIRMED_PENDING_HUMAN_REVIEW":
        return "Human review required";

      case "CONFIRMED_PENDING_VERIFICATION":
        return "Additional verification required";

      case "ESCALATED_FOR_INVESTIGATION":
        return "Escalated for investigation";

      case "REVIEW_CLOSED_NO_BANKING_ACTION":
        return "Review closed";

      default:
        return status;
    }
  }

  function outcomeMessage() {
    switch (status) {
      case "DISPUTED":
        return (
          "The customer denied the request. " +
          "It must not proceed without investigation."
        );

      case "ESCALATED_FOR_INVESTIGATION":
        return (
          "The simulated officer escalated this case " +
          "for investigation. No banking action occurs."
        );

      case "REVIEW_CLOSED_NO_BANKING_ACTION":
        return (
          "The simulated officer closed the review " +
          "without performing a banking operation."
        );

      default:
        return (
          "The customer confirmed the request, " +
          "but further verification or review is " +
          "still required."
        );
    }
  }

  return (
    <>
      <section className="panel verification-panel">
        <div className="panel-heading">
          <h3>Banking Verification Simulator</h3>
          <span className="step">
            04 / VERIFICATION
          </span>
        </div>

        <p className="verification-description">
          Create a simulated banking verification
          case, record a response and retrieve its
          history from SQLite.
        </p>

        {analysis &&
          needsVerification &&
          !caseRecord && (
            <div className="verification-phone">
              <div className="verification-app-header">
                <span className="verification-app-icon">
                  B
                </span>

                <div>
                  <strong>
                    Banking Security Console
                  </strong>

                  <small>
                    Case creation
                  </small>
                </div>
              </div>

              <h4>
                Additional verification required
              </h4>

              <p>
                Requested banking operation:
              </p>

              <div className="verification-request">
                {actionNames[
                  context.requested_action
                ] || context.requested_action}
              </div>

              <p>
                Policy risk score:{" "}
                {risk.risk_score}/100
              </p>

              <button
                type="button"
                className="verification-approve"
                style={{
                  width: "100%",
                  border: 0,
                  borderRadius: 8,
                  padding: 13,
                  fontWeight: 700,
                  cursor: "pointer",
                }}
                disabled={loading}
                onClick={createCase}
              >
                {loading
                  ? "Creating Case..."
                  : "Create Verification Case"}
              </button>
            </div>
          )}

        {analysis &&
          !needsVerification && (
            <p className="verification-disclaimer">
              The selected scenario follows the
              standard banking flow. No additional
              verification case is needed.
            </p>
          )}

        {!analysis && !caseRecord && (
          <p className="verification-disclaimer">
            Run a call analysis to create a new
            verification case, or load an existing
            case below.
          </p>
        )}

        {caseRecord && (
          <div className="verification-phone">
            <div className="verification-app-header">
              <span className="verification-app-icon">
                B
              </span>

              <div>
                <strong>
                  Banking App
                </strong>

                <small>
                  Simulated customer confirmation
                </small>
              </div>
            </div>

            <h4>
              Confirm your request
            </h4>

            <p>
              Banking operation:
            </p>

            <div className="verification-request">
              {actionNames[
                caseRecord.banking_context.banking_action
              ] ||
                caseRecord.banking_context
                  .banking_action}
            </div>

            <p style={{ overflowWrap: "anywhere" }}>
              <strong>Case ID:</strong>{" "}
              {caseRecord.case_id}
            </p>

            <p>
              <strong>Status:</strong>{" "}
              {status}
            </p>

            {status ===
            "PENDING_CUSTOMER_RESPONSE" ? (
              <>
                <p>
                  Did you initiate this banking
                  request?
                </p>

                <div className="verification-actions">
                  <button
                    type="button"
                    className="verification-approve"
                    disabled={loading}
                    onClick={() =>
                      respond("confirm")
                    }
                  >
                    Yes, it's me
                  </button>

                  <button
                    type="button"
                    className="verification-deny"
                    disabled={loading}
                    onClick={() =>
                      respond("deny")
                    }
                  >
                    No, it's not me
                  </button>
                </div>
              </>
            ) : (
              <div
                className={`verification-outcome ${outcomeTone}`}
                role="status"
              >
                <strong>
                  {outcomeTitle()}
                </strong>

                <p>
                  {outcomeMessage()}
                </p>
              </div>
            )}

            <h4>
              Audit History
            </h4>

            <ul style={{ paddingLeft: 20 }}>
              {caseRecord.audit_events.map(
                (event) => (
                  <li
                    key={event.event_id}
                    style={{
                      marginBottom: 8,
                      fontSize: 12,
                      overflowWrap: "anywhere",
                    }}
                  >
                    <strong>
                      {event.event_type}
                    </strong>

                    <div>
                      {event.timestamp}
                    </div>
                  </li>
                )
              )}
            </ul>
          </div>
        )}

        <div
          style={{
            maxWidth: 440,
            marginTop: 22,
          }}
        >
          <label
            htmlFor="verificationCaseId"
            className="field-label"
          >
            Load Existing Verification Case
          </label>

          <input
            id="verificationCaseId"
            value={lookupId}
            onChange={(event) =>
              setLookupId(event.target.value)
            }
            placeholder="Enter case ID"
            style={{
              width: "100%",
              background: "#0d1729",
              border: "1px solid #344661",
              padding: 12,
              borderRadius: 8,
              color: "#eef2ff",
            }}
          />

          <button
            type="button"
            className="analyze-button"
            style={{ marginTop: 10 }}
            disabled={loading}
            onClick={loadCase}
          >
            {loading
              ? "Please Wait..."
              : "Load Case"}
          </button>
        </div>

        {error && (
          <div
            className="error-box"
            role="alert"
          >
            {error}
          </div>
        )}

        {message && (
          <p
            role="status"
            style={{
              color: "#86efac",
              fontSize: 13,
              marginTop: 15,
            }}
          >
            {message}
          </p>
        )}

        <p className="verification-disclaimer">
          Local demo only. These responses are
          not authenticated banking-app
          notifications. No actual transaction,
          customer verification or account change
          is performed.
        </p>
      </section>

      {/* PHASE 9: CASE DASHBOARD */}
      <CaseDashboard
        refreshToken={refreshToken}
        onCaseUpdated={
          handleDashboardCaseUpdated
        }
      />
    </>
  );
}
