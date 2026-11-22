
import { useEffect, useState } from "react";
import "./CaseDashboard.css";

const API_URL = "http://127.0.0.1:8000";
const PAGE_SIZE = 10;

const statusLabels = {
  PENDING_CUSTOMER_RESPONSE: "Pending Response",
  CONFIRMED_PENDING_VERIFICATION: "Pending Verification",
  CONFIRMED_PENDING_HUMAN_REVIEW: "Pending Human Review",
  DISPUTED: "Disputed",
  ESCALATED_FOR_INVESTIGATION: "Escalated",
  REVIEW_CLOSED_NO_BANKING_ACTION: "Review Closed",
};

const actionLabels = {
  balance_inquiry: "Balance Enquiry",
  transaction_status: "Transaction Status",
  password_reset: "Password Reset",
  mobile_number_change: "Mobile Number Change",
  transfer_limit_increase: "Transfer Limit Increase",
  high_value_transfer: "High Value Transfer",
};

function formatTime(value) {
  if (!value) return "—";
  return new Date(value).toLocaleString();
}

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
        : `HTTP ${response.status}`
    );
  }

  return data;
}

export default function CaseDashboard({
  refreshToken = 0,
  onCaseUpdated,
}) {
  const [statusFilter, setStatusFilter] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [appliedSearch, setAppliedSearch] = useState("");

  const [page, setPage] = useState(0);
  const [refreshIndex, setRefreshIndex] = useState(0);

  const [cases, setCases] = useState([]);
  const [counts, setCounts] = useState({});
  const [total, setTotal] = useState(0);

  const [selectedCase, setSelectedCase] = useState(null);

  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [reviewLoading, setReviewLoading] = useState(false);

  const [error, setError] = useState("");
  const [reviewError, setReviewError] = useState("");
  const [message, setMessage] = useState("");

  const [reviewNote, setReviewNote] = useState("");

  // Fetch cases whenever a filter, page or
  // parent component's data changes.
  useEffect(() => {
    const controller = new AbortController();

    async function loadCases() {
      setLoading(true);
      setError("");

      const params = new URLSearchParams({
        limit: String(PAGE_SIZE),
        offset: String(page * PAGE_SIZE),
      });

      if (statusFilter) {
        params.set("status", statusFilter);
      }

      if (appliedSearch) {
        params.set("search", appliedSearch);
      }

      try {
        const response = await fetch(
          `${API_URL}/api/verification-cases?${params}`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error(
            `Unable to load cases (HTTP ${response.status})`
          );
        }

        const data = await response.json();

        if (!controller.signal.aborted) {
          setCases(data.items);
          setCounts(data.status_counts);
          setTotal(data.total);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err.message);
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    loadCases();

    return () => controller.abort();
  }, [
    statusFilter,
    appliedSearch,
    page,
    refreshIndex,
    refreshToken,
  ]);

  function refreshCases() {
    setRefreshIndex((value) => value + 1);
  }

  function applySearch(event) {
    event.preventDefault();
    setPage(0);
    setAppliedSearch(searchInput.trim());
  }

  async function viewCase(caseId) {
    setDetailLoading(true);
    setError("");
    setReviewError("");
    setMessage("");

    try {
      const data = await apiRequest(
        `/api/verification-cases/${caseId}`
      );

      setSelectedCase(data);
      setReviewNote("");
    } catch (err) {
      setError(err.message);
    } finally {
      setDetailLoading(false);
    }
  }

  async function submitReview(decision) {
    if (!selectedCase || reviewLoading) return;

    const note = reviewNote.trim();

    if (note.length < 8) {
      setReviewError(
        "Please enter a review note of at least 8 characters."
      );
      return;
    }

    setReviewLoading(true);
    setReviewError("");
    setMessage("");

    try {
      const updatedCase = await apiRequest(
        `/api/verification-cases/${selectedCase.case_id}/review`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            decision,
            note,
          }),
        }
      );

      setSelectedCase(updatedCase);
      setReviewNote("");
      setMessage(
        "Simulated officer review saved to SQLite."
      );

      onCaseUpdated?.(updatedCase);
      refreshCases();
    } catch (err) {
      setReviewError(err.message);
    } finally {
      setReviewLoading(false);
    }
  }

  const pending = counts.PENDING_CUSTOMER_RESPONSE || 0;

  const confirmed =
    (counts.CONFIRMED_PENDING_VERIFICATION || 0) +
    (counts.CONFIRMED_PENDING_HUMAN_REVIEW || 0);

  const disputed = counts.DISPUTED || 0;

  const escalated =
    counts.ESCALATED_FOR_INVESTIGATION || 0;

  const allCases = Object.values(counts).reduce(
    (sum, count) => sum + count,
    0
  );

  const canReview =
    selectedCase &&
    [
      "DISPUTED",
      "CONFIRMED_PENDING_HUMAN_REVIEW",
      "ESCALATED_FOR_INVESTIGATION",
    ].includes(selectedCase.status);

  const canEscalate =
    canReview &&
    selectedCase.status !==
      "ESCALATED_FOR_INVESTIGATION";

  const totalPages = Math.max(
    1,
    Math.ceil(total / PAGE_SIZE)
  );

  return (
    <section className="panel case-dashboard">
      <div className="panel-heading">
        <h3>Fraud Case Management</h3>
        <span className="step">05 / CASE DASHBOARD</span>
      </div>

      <p className="case-description">
        Browse simulated verification cases, inspect
        audit history and record review outcomes.
      </p>

      <div className="case-summary">
        <div className="case-stat">
          <span>All Cases</span>
          <strong>{allCases}</strong>
        </div>

        <div className="case-stat">
          <span>Pending</span>
          <strong>{pending}</strong>
        </div>

        <div className="case-stat">
          <span>Confirmed</span>
          <strong>{confirmed}</strong>
        </div>

        <div className="case-stat">
          <span>Disputed</span>
          <strong>{disputed}</strong>
        </div>

        <div className="case-stat">
          <span>Escalated</span>
          <strong>{escalated}</strong>
        </div>
      </div>

      <div className="case-toolbar">
        <form
          className="case-search"
          onSubmit={applySearch}
        >
          <input
            type="text"
            value={searchInput}
            maxLength={100}
            placeholder="Search case ID or banking action"
            onChange={(event) =>
              setSearchInput(event.target.value)
            }
          />

          <button type="submit">
            Search
          </button>
        </form>

        <select
          aria-label="Filter case status"
          value={statusFilter}
          onChange={(event) => {
            setStatusFilter(event.target.value);
            setPage(0);
          }}
        >
          <option value="">All Statuses</option>

          {Object.entries(statusLabels).map(
            ([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            )
          )}
        </select>

        <button
          type="button"
          className="case-refresh"
          onClick={refreshCases}
        >
          Refresh
        </button>
      </div>

      {error && (
        <div className="error-box" role="alert">
          {error}
        </div>
      )}

      <div className="case-table-scroll">
        <table className="case-table">
          <thead>
            <tr>
              <th>Case ID</th>
              <th>Banking Operation</th>
              <th>Risk</th>
              <th>Status</th>
              <th>Updated</th>
              <th>Details</th>
            </tr>
          </thead>

          <tbody>
            {cases.map((item) => (
              <tr key={item.case_id}>
                <td
                  title={item.case_id}
                  className="case-id"
                >
                  {item.case_id.slice(0, 10)}...
                </td>

                <td>
                  {actionLabels[item.banking_action] ||
                    item.banking_action}
                </td>

                <td>
                  {item.risk_score}/100
                </td>

                <td>
                  <span
                    className="case-status"
                    data-status={item.status}
                  >
                    {statusLabels[item.status] ||
                      item.status}
                  </span>
                </td>

                <td>
                  {formatTime(item.updated_at)}
                </td>

                <td>
                  <button
                    type="button"
                    className="case-view"
                    disabled={detailLoading}
                    onClick={() =>
                      viewCase(item.case_id)
                    }
                  >
                    View
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {!loading && cases.length === 0 && (
          <p className="case-empty">
            No verification cases match these filters.
          </p>
        )}
      </div>

      <div className="case-pagination">
        <span>
          {loading
            ? "Loading cases..."
            : `${total} matching case(s) · Page ${
                page + 1
              } of ${totalPages}`}
        </span>

        <div>
          <button
            type="button"
            disabled={page === 0 || loading}
            onClick={() =>
              setPage((value) => Math.max(0, value - 1))
            }
          >
            Previous
          </button>

          <button
            type="button"
            disabled={
              page + 1 >= totalPages || loading
            }
            onClick={() =>
              setPage((value) => value + 1)
            }
          >
            Next
          </button>
        </div>
      </div>

      {selectedCase && (
        <div className="case-details">
          <div className="case-details-heading">
            <h4>Case Investigation Details</h4>

            <button
              type="button"
              onClick={() => setSelectedCase(null)}
              className="case-refresh"
            >
              Close
            </button>
          </div>

          <p className="case-full-id">
            <strong>Case ID:</strong>{" "}
            {selectedCase.case_id}
          </p>

          <p>
            <strong>Operation:</strong>{" "}
            {actionLabels[
              selectedCase.banking_context.banking_action
            ]}
          </p>

          <p>
            <strong>Authentication:</strong>{" "}
            {selectedCase.banking_context
              .authentication_status}
          </p>

          <p>
            <strong>Risk Score:</strong>{" "}
            {selectedCase.risk_assessment.risk_score}/100
          </p>

          <p>
            <strong>Status:</strong>{" "}
            {statusLabels[selectedCase.status] ||
              selectedCase.status}
          </p>

          <h4>Risk Reasons</h4>

          <ul>
            {selectedCase.risk_assessment.reasons.map(
              (reason, index) => (
                <li key={index}>{reason}</li>
              )
            )}
          </ul>

          <h4>Audit History</h4>

          <div className="case-audit">
            {selectedCase.audit_events.map((event) => (
              <div key={event.event_id}>
                <strong>{event.event_type}</strong>
                <small>
                  {formatTime(event.timestamp)}
                </small>
              </div>
            ))}
          </div>

          {selectedCase.officer_reviews?.length > 0 && (
            <>
              <h4>Recorded Officer Reviews</h4>

              {selectedCase.officer_reviews.map(
                (review) => (
                  <div
                    className="case-review-record"
                    key={review.review_id}
                  >
                    <strong>
                      {review.decision === "escalate"
                        ? "Escalated"
                        : "Review closed without banking action"}
                    </strong>

                    <p>{review.note}</p>

                    <small>
                      {formatTime(review.timestamp)}
                    </small>
                  </div>
                )
              )}
            </>
          )}

          {canReview && (
            <div className="case-review-panel">
              <h4>Simulated Officer Review</h4>

              <p>
                Enter an investigation note and
                select a review outcome. Do not enter
                real customer or account details.
              </p>

              <label
                htmlFor="caseReviewNote"
                className="field-label"
              >
                Review Note
              </label>

              <textarea
                id="caseReviewNote"
                value={reviewNote}
                maxLength={500}
                rows={4}
                placeholder="Example: Customer denied the request. Escalating for manual investigation."
                onChange={(event) =>
                  setReviewNote(event.target.value)
                }
              />

              <div className="case-review-buttons">
                {canEscalate && (
                  <button
                    type="button"
                    className="case-escalate"
                    disabled={reviewLoading}
                    onClick={() =>
                      submitReview("escalate")
                    }
                  >
                    Escalate Case
                  </button>
                )}

                <button
                  type="button"
                  className="case-close-review"
                  disabled={reviewLoading}
                  onClick={() =>
                    submitReview("close_no_action")
                  }
                >
                  Close Review — No Banking Action
                </button>
              </div>

              {reviewError && (
                <div className="error-box" role="alert">
                  {reviewError}
                </div>
              )}

              {message && (
                <p className="case-success" role="status">
                  {message}
                </p>
              )}
            </div>
          )}

          {!canReview && (
            <p className="case-description">
              No officer-review action is currently
              available for this case status.
            </p>
          )}
        </div>
      )}

      <p className="case-safety-note">
        Simulated case management only. Review buttons
        do not authorize transactions, establish customer
        identity, freeze accounts or contact a real bank.
      </p>
    </section>
  );
}
